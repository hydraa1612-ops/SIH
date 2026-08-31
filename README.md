## 1. The Problem Statement, in full

**PS #88 — Battery Health Monitoring & Remaining Useful Life Prediction**
Sector: EV | Category: Software

**As written:** "Battery degradation significantly affects EV performance and replacement costs. Develop AI models capable of continuously estimating battery State of Health (SoH), remaining useful life and early fault detection using BMS and sensor telemetry."

**Expected solution (as written):** "Predictive battery analytics dashboard and maintenance recommendations."

### What this is actually asking for, in plain terms

Every EV battery slowly loses capacity as it's used — this is normal, unavoidable, and happens to every lithium-ion battery on earth. The problem is that right now, most people (whether that's an individual EV owner or a company running a fleet of delivery vehicles) have almost no visibility into how far along that degradation actually is. They find out the hard way: the car starts showing less range than it used to, or worse, a battery fails unexpectedly.

The PS is asking for three connected things:

1. **State of Health (SOH)** — a live answer to "how much of this battery's original capacity is actually left, right now, as a percentage."
2. **Remaining Useful Life (RUL)** — a forward-looking answer to "roughly how much longer, or how many more charge cycles, until this battery hits a point where it should probably be serviced or replaced."
3. **Early fault detection** — catching something going wrong (like unusual internal resistance, or a fault developing in one part of the pack) before it turns into an actual failure or safety issue.

And it wants all of this built from actual **BMS (Battery Management System) and sensor telemetry** — meaning the data a real EV's onboard systems already collect, not something invented or assumed.

### Why this problem actually matters

Two costs are named directly in the PS, and they're real, well-documented pain points in the EV industry:

- **Unplanned downtime.** If a fleet vehicle's battery fails without warning, that's a vehicle out of service, a delivery missed, a driver stranded. For a company running dozens or hundreds of EVs, this adds up fast.
- **Premature replacement.** EV batteries are one of the most expensive single components in the vehicle — often 30-40% of the total vehicle cost. Without a reliable way to know a battery's actual remaining health, the safe default is to replace it earlier than necessary "just in case," which wastes money and, from an environmental angle, wastes a battery that still had useful life left.

There's also a slower-moving, less obvious cost: **trust**. A lot of hesitation around buying an EV, especially in a market like India where EV adoption is still ramping up, comes down to uncertainty about battery longevity. A system that can actually show a real, honest answer to "how healthy is my battery" chips away at that uncertainty.

---

## 2. How CellSense actually works

CellSense isn't one single model — it's a small pipeline of three connected predictions, sitting on top of a deliberately careful data foundation, with a plain-language layer on top that turns predictions into something a person can actually act on.

### Step 1: Getting real data, and not pretending different things are the same thing

We use four real, publicly available datasets, and this is the part we're most careful about, because it's also the part most similar projects get sloppy on.

- **A real field dataset from 20 electric vehicles**, tracked for about 29 months, collecting actual charging data (voltage, current, and a monthly capacity reading calculated the standard way battery researchers do it — the Ampere-integral method). This is the closest thing we have to "real cars, real driving, real charging," and it's our main source for the SOH model.
- **NASA's Li-ion battery aging dataset** — individual lab cells that were deliberately charged and discharged over and over until they reached a documented end-of-life point (typically defined as 30% capacity fade). This gives us clean, controlled data with a very clear "this is where the battery died" marker, which is genuinely useful for the RUL model, even though it's lab cells and not full EV packs.
- **BatteryLife**, a large research dataset covering nearly a thousand batteries across many different chemistries and formats. We use this mainly as a cross-check — does our model still make sense on batteries it's never seen before, built differently, from a different source?
- **A fault-labeled dataset (CH-BatteryGen)** built from real EV operating data, with clearly labeled fault examples (self-discharge, high internal resistance, low capacity) generated on top of that real data using rules grounded in actual battery-fault mechanisms. This is our main source for training the fault-detection model, since real, clearly-labeled fault events are genuinely rare to find in raw data at any usable scale.

The important discipline here: **a single lab cell's capacity and a full EV battery pack's capacity are not the same physical quantity.** Before any of these four sources touch a model, they're all converted into the same format — normalized to State of Health as a percentage, tagged with exactly which dataset they came from, and never silently combined as if they were interchangeable. If we ever show a number to a judge or a user, we can say exactly which real dataset it came from.

### Step 2: Three models, each doing one job

- **SOH model** takes charging-session data (average voltage during charge, how long the charge took, current patterns, temperature) and estimates the battery's current health as a percentage. Trained and tested mainly on the real 20-EV data, with the model's performance checked separately (not blended) against the lab and multi-chemistry data as a robustness sanity check.
- **RUL model** looks at how a battery's capacity has been declining over time and projects forward to estimate how much time or how many cycles remain before it crosses a defined "needs attention" threshold. Built primarily on NASA's data, since that's the source with real documented run-to-failure information.
- **Fault model** classifies current battery behavior into normal, or one of a few known fault categories, based on patterns in voltage, current, and temperature that don't look like healthy operation. Trained on the fault-labeled dataset, then run against real (unlabeled) EV data as a sanity check — flagging things that look unusual, without pretending we've "proven" fault-detection accuracy on data that was never labeled in the first place.

### Step 3: Turning three predictions into one clear recommendation

This is the part we think matters most, and it's deliberately the simplest part, technically. Instead of a fourth machine learning model trying to combine everything (which would just be another black box stacked on top of three other models), we use a small set of plain if-then rules. Something like: if the fault model flags something abnormal, say so directly. If SOH has dropped below a set threshold, recommend inspection. If none of that applies but the model's confidence in its own prediction is low, say that too, instead of pretending to be certain when it isn't.

That last part matters a lot for something like this. A model that's uncertain but reports a confident-sounding number is arguably more dangerous than one that just says "not sure, check this manually."

### Step 4: A dashboard that shows the real thing, not a mockup

The final layer is a simple web dashboard: pick a battery, see its SOH trend over time, see the RUL estimate with the threshold it's measured against clearly stated, see the raw signals if you want to dig in, see any fault flags, and see the plain-language maintenance recommendation. Every number on that screen traces back to something the models actually computed from real data — nothing is hardcoded for the sake of the demo looking good.

---

## 3. What it's built with, and how the pieces fit together

Everything in this stack is free-tier, on purpose, since this needs to run on a student team's actual laptops and a college hackathon budget of essentially zero.

**Data layer.** Python, with Polars and DuckDB doing the heavy lifting of reading and filtering large data files without needing to load everything into memory at once (this matters because one of our four datasets is several gigabytes, and not every team laptop has a lot of RAM). Processed data is stored as small Parquet files, not raw CSVs, because Parquet is much faster to read and takes up far less space.

**Modeling layer.** scikit-learn for simple baseline models first (so we always have something working and interpretable to compare against), then LightGBM as the stronger model for each of the three prediction tasks. All of this runs on ordinary CPUs — nothing here needs a GPU, given the actual size of the data involved.

**Backend.** FastAPI, written in Python. This matters more than it sounds like it should: because our models are trained in Python, the backend that serves predictions is also Python, so there's no awkward handoff between two different programming languages just to get a prediction from a trained model into an API response. It keeps the whole system simpler and gives us fewer places for something to quietly break.

**Frontend.** React with Tailwind CSS for styling and Recharts for the graphs (SOH trend lines, confidence bands, raw signal plots).

**Hosting.** Vercel for the frontend, Render for the backend API, both on their free tiers. If we ever need to persist data across sessions, Supabase's free tier is available, though the core system doesn't strictly need a database to demonstrate the main idea.

**How it all connects, start to finish:**

Raw datasets get downloaded once → each one runs through its own small "adapter" script that converts it into our shared data format → a quality-control pass checks for obviously broken or missing values → the cleaned data trains the three models → those three models' outputs feed into the plain-rules maintenance layer → the API serves all of this to the dashboard, where a person can actually look at it.

One detail that matters technically: when we split data for training versus testing, we split by **which battery or vehicle** the data came from, never by individual data rows. If we didn't do this, the model could accidentally "see" a battery during training and then get tested on more data from that same battery, which would make it look far more accurate than it actually is on a battery it's never encountered before. This is a well-known trap in this kind of work, and it's one we specifically built our process around avoiding.

---

## 4. Feasibility

**Data feasibility.** All four datasets are real, already public, and we've confirmed their actual sizes are workable on the hardware our team has (one 24GB laptop, three 16GB, one 8GB, one 4GB). The largest dataset gets processed once, on the machine with the most memory, and everyone else works from the small cleaned files it produces — nobody else needs to touch the multi-gigabyte raw files directly.

**Technical feasibility.** Nothing in this pipeline requires anything unusual. Gradient-boosted models on tabular data are a well-established, well-understood approach — this isn't cutting-edge research, it's solid, dependable engineering applied carefully to real data. That's a deliberate choice: the goal isn't to build something flashy that might not work, it's to build something that reliably does what it says.

**Team feasibility.** We've split responsibilities across the pipeline so no single person is a bottleneck — someone owns the data/schema work, someone owns the three models, others handle backend, frontend, and pulling everything together for the actual demo, so different parts of the build can happen in parallel instead of waiting on each other.

**Cost feasibility.** Zero. Every tool in the stack is free-tier or fully open source. There's no dependency on anything that could get cut off mid-hackathon because a free trial ran out or a card needs to be on file somewhere.

---

## 5. The real production challenges, and what we actually did about each one

We think being upfront about these matters more than pretending they don't exist — a panel that's paying attention will find the gaps anyway, so we'd rather show we already found them ourselves.

### Challenge 1: No dataset gives us "the real thing" exactly as the PS describes it

The PS asks for BMS and sensor telemetry from real EVs. What's actually publicly available is close, but not identical — our best real-vehicle dataset is charging data specifically, not full driving-cycle telemetry, and it's from one vehicle model and one battery chemistry.

**What we did:** we said so, clearly, instead of implying our data is something it isn't. We use it for exactly what it's good for (SOH from charging behavior) and we don't stretch a claim about it further than the data supports. If asked to generalize across every EV on the road, our honest answer is that we've validated on one real fleet and multiple lab benchmarks, and that broader generalization is a real, acknowledged limitation, not a solved problem.

### Challenge 2: We don't know yet whether our RUL model has real degradation events to learn from

RUL prediction needs batteries that actually get close to a meaningful end-of-life point in the data. Our real 20-EV dataset only covers 29 months, and EV batteries commonly last considerably longer than that before serious degradation sets in. It's genuinely possible that none of those 20 vehicles crossed a meaningful degradation threshold in the time they were recorded.

**What we did:** built this as an explicit first check, before writing deeper model code — plot every vehicle's capacity trend and see how close any of them actually get to a real threshold. If the real data doesn't support it, our RUL claims lean more heavily on NASA's lab data, which does have real documented failure points, and we say plainly which of our two data sources is doing the work for that specific claim.

### Challenge 3: Our fault-detection data isn't raw, unedited real-world fault recordings

Real, clearly-labeled fault events are hard to find in raw data at any scale that's useful for training a model. The dataset we use is built from real EV operating data, with fault examples generated on top of it using rules based on how faults actually behave physically — not fabricated from nothing, but also not the same as pulling unedited fault events straight out of a fleet's logs.

**What we did:** we describe this dataset accurately rather than either overselling it ("real-world fault data") or underselling it ("just synthetic data") — it's real operational data with mechanism-grounded fault generation layered on top, and we say that exact thing if asked. We also only use it for training the classifier, and treat any flags it raises on real, unlabeled EV data as a sanity check, not as proof the fault detector is validated in the real world.

### Challenge 4: Keeping a multi-source pipeline honest under time pressure

With four datasets, three models, and a small team, there's a real risk of quietly cutting corners — merging things that shouldn't be merged, skipping the leakage-safe split because it's slower, or letting a confident-sounding number slip into the pitch without checking it's actually grounded in something real.

**What we did:** built explicit rules into our own process from the start rather than relying on remembering to be careful later — a fixed shared data format every source has to go through, mandatory quality checks before any data touches a model, always splitting by battery/vehicle rather than by row, and a running list of claims we've specifically decided not to make (like implying live BMS integration, or claiming exact cost savings we don't have evidence for).

### Challenge 5: Hardware differences across the team

Not every laptop on the team can handle the same workload — one has 4GB of RAM, which genuinely can't run a code editor, a local server, and a browser all at once without risking a crash.

**What we did:** matched work to hardware honestly instead of hoping it would somehow be fine. The heaviest one-time data processing happens once, on the strongest machine, and everyone else — including the lightest laptop — only ever works from small, already-cleaned files. The lightest machine is scoped to documentation, testing the live deployed app, and demo-day support, not local development.

---

## 6. Real-world impact, across every field it touches

**For EV fleet operators and logistics companies.** This is probably the most direct beneficiary. A delivery or ride-hailing fleet running dozens of EVs can move from reactive maintenance (wait for something to break) to planned maintenance (know in advance which vehicles need attention soon), which directly reduces missed deliveries, stranded drivers, and emergency repair costs.

**For EV service and maintenance centers.** Instead of a technician doing a manual, time-consuming diagnostic on every vehicle that comes in, a system like this gives them a starting point — which vehicles actually need deeper inspection, and roughly why, before they even open the hood.

**For individual EV owners, eventually.** Right now most consumer EVs give very little visibility into real battery health beyond a rough range estimate. Something like CellSense, integrated further down the line, could give an owner an honest, ongoing answer instead of finding out only when something goes wrong.

**Economically.** Two costs, directly targeted: fewer batteries replaced earlier than they need to be, and less money lost to unplanned vehicle downtime. Both are concrete, measurable savings for anyone operating EVs at any scale.

**Environmentally.** A battery that's replaced only when it actually needs to be, instead of "just in case," is a battery that isn't prematurely turned into waste. EV batteries are resource-intensive to manufacture (lithium, cobalt, nickel extraction all carry real environmental costs), so extending genuine usable life has a real environmental upside, not just a cost-saving one.

**Socially, and specifically in the Indian context.** EV adoption is growing fast in India, but battery longevity and reliability remain a real point of hesitation for a lot of potential buyers. A transparent, trustworthy way to monitor battery health chips away at that uncertainty — not by making unrealistic promises, but by actually showing people honest information instead of a black box.

**For the secondhand EV market.** This is a smaller but genuinely underserved angle: buying a used EV right now is risky specifically because battery health is so hard to verify independently. A system that can give a credible, data-backed SOH reading could eventually support a more trustworthy resale market, similar to how an odometer or a vehicle history report works for conventional cars.

**For manufacturers and BMS developers, longer-term.** While our prototype doesn't integrate with live proprietary BMS systems (because that access simply isn't available to us), the architecture is built so that a real BMS data feed could be integrated later without redesigning the whole system. The ingestion layer is built to accept that kind of input; it's just not populated with real live data in this hackathon version, and we say that clearly rather than imply otherwise.
If there's one sentence that summarizes the whole approach: **we tried to build the simplest version of this system that we could fully stand behind, rather than the most impressive-sounding version we couldn't.**

**NEW UPDATES:**
# ⚡ EV Battery Management System (BMS) Diagnostics & Prognostics Platform

An end-to-end Machine Learning and REST API platform for real-time EV battery telemetry monitoring, State of Health ($\text{SOH}\%$) estimation, multi-class fault classification, and Remaining Useful Life ($\text{RUL}$) cycle prediction.

---

## 📌 Project Architecture
                              +-----------------------+
                              |   Streamlit Frontend  |
                              |   (src/dashboard)     |
                              +-----------+-----------+
                                          |
                                 HTTP POST | JSON Payload
                                          v
                              +-----------------------+
                              |    FastAPI REST Service|
                              |    (src/api/app.py)    |
                              +-----------+-----------+
                                          |
                                          | Feature Processing
                                          v
  +---------------------------------------+---------------------------------------+
  |                                       |                                       |
  v                                       v                                       v
+-----------------------+           +-----------------------+           +-----------------------+
|  Scale-Invariant SOH  |           |  Fault Classifier     |           |  RUL Prognostics      |
|  LightGBM Regressor   |           |  Multi-class LightGBM |           |  LightGBM Regressor   |
+-----------------------+           +-----------------------+           +-----------------------+

---

## 📊 ML Model Performance & Specifications

| Model Target | Algorithm | Key Input Features | Primary Metrics |
| :--- | :--- | :--- | :--- |
| **State of Health (SOH%)** | LightGBM Regressor | `cell_voltage_avg`, `cell_voltage_min`, `cell_voltage_max`, `temperature`, `SOC` | **MAE:** $1.63\%$<br>**$R^2$ Score:** $0.9901$ |
| **Fault Status** | LightGBM Classifier | `voltage`, `current`, `temperature`, `SOC`, `cell_voltage_min`, `cell_voltage_max` | **Accuracy:** $> 98\%$<br>**Classes:** Normal, Cell Imbalance, Thermal Anomaly |
| **Remaining Useful Life (RUL)** | LightGBM Regressor | `voltage`, `current`, `temperature`, `SOC`, `cell_voltage_min`, `cell_voltage_max` | **MAE:** $\sim 12$ Cycles |

---

## 📁 Repository Structure

SIH/
├── data/                      # Raw and processed battery parquet datasets
├── notebooks/                 # Exploratory data analysis & model development notebooks
├── saved_models/              # Trained joblib artifacts (.pkl files)
├── src/
│   ├── api/                   # FastAPI backend implementation
│   │   └── app.py
│   ├── dashboard/             # Interactive Streamlit visualizer
│   │   └── app.py
│   ├── features/              # Feature engineering modules
│   ├── models/                # Model training and cross-check scripts
│   └── pipeline/              # Rules engine and safety logic
├── requirements.txt           # Python environment dependencies
└── README.md                  # Project documentation
---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```powershell
# Clone repository
git clone [https://github.com/hydraa1612-ops/SIH.git](https://github.com/hydraa1612-ops/SIH.git)
cd SIH

# Create and activate virtual environment
python -m venv astro_env
.\astro_env\Scripts\Activate.ps1

# Install requirements
pip install -r backend/requirements.txt

2. Launch FastAPI Backend
Start the high-performance inference REST API:
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

Interactive Swagger Docs available at: http://127.0.0.1:8000/docs

3. Launch Streamlit UI Dashboard
In a separate terminal window, start the frontend interface:

PowerShell
streamlit run src/dashboard/app.py
Dashboard URL: http://localhost:8501

🔌 API Endpoint Reference
POST /predict
Runs unified batch inference across all three ML diagnostic models.

Sample Request Body:

JSON
{
  "voltage": 115.2,
  "current": -12.5,
  "temperature": 28.5,
  "SOC": 82.0,
  "cell_voltage_min": 3.58,
  "cell_voltage_max": 3.62
}
Sample Response Body:

JSON
{
  "soh_percentage": 80.5,
  "fault_status": "NORMAL",
  "estimated_rul_cycles": 607,
  "telemetry_received": { ... }
}

---

**Git Terminal Commands**

Run these in your VS Code terminal after updating and saving `README.md`:

```powershell
git add README.md
git commit -m "Docs: Update README with project architecture and API reference"
git push

