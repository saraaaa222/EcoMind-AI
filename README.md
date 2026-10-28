<div align="center">

# ⚡ EcoMind AI
### Intelligent Home Energy Optimization Using LSTM Forecasting and Fuzzy Logic

EcoMind AI is an academic prototype exploring how deep learning and fuzzy logic can support smarter home-energy analysis. It combines LSTM forecasting models, a fuzzy decision system, and a simulation-based energy optimizer in an interactive Streamlit dashboard.

> **Project status:** Academic prototype. The project does not currently control real household appliances, batteries, or electrical equipment.

## ✨ Features

- **Consumption forecasting:** LSTM model using a 24-hour historical input window.
- **Solar forecasting:** LSTM model using a 168-hour historical input window.
- **Fuzzy decision engine:** Produces illustrative recommendations for appliance reduction, battery action, and grid interaction.
- **Interactive dashboard:** Overview, Predictions, AI Decisions, and Analytics tabs.
- **Energy optimization simulation:** Compares baseline grid demand with a heuristic demand-shifting scenario.
- **Scenario analytics:** Shows grid-demand changes and illustrative costs using an assumed tariff.

## 🏗️ System Architecture

1. **Data processing:** Load and align hourly consumption and solar datasets.
2. **LSTM models:** Process historical feature windows to produce next-step predictions.
3. **Fuzzy logic:** Evaluate normalized energy-related inputs and generate decision outputs.
4. **Energy simulation:** Shift a configurable share of demand toward hours with solar surplus.
5. **Dashboard:** Visualize predictions, decisions, historical patterns, and scenario comparisons.

**Important:** The analytics simulator currently operates separately from the LSTM forecasts. The dashboard is a simulation, not a verified real-time energy-control system.

## 📁 Project Structure

```text
Smart-Home-energy-consumption/
├── dashboard.py              # Streamlit dashboard
├── energy_optimizer.py       # Energy data processing and simulation
├── consumption_model.py      # Consumption LSTM training
├── solar_model.py            # Solar-generation LSTM training
├── fuzzy_system.py           # Fuzzy decision system
├── test_fuzzy.py             # Fuzzy-system tests
├── consumption_data.csv      # Hourly consumption dataset
├── solar_data.csv            # Hourly solar dataset
├── consumption_model.h5      # Saved consumption model
├── solar_model.h5            # Saved solar model
└── README.md                 # Project documentation
```

## 🧠 AI Models

### 1. Consumption LSTM

| Parameter | Details |
|---|---|
| Input window | 24 hourly steps |
| Input features | 9 |
| Output | One next-step prediction |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Training approach | Sequential train/validation/test split |

**Features:** `consumption`, `temperature`, `hour`, `day_of_week`, `month`, `occupancy`, `weather`, `previous_day_cons`, `weekend`.

### 2. Solar-Generation LSTM

| Parameter | Details |
|---|---|
| Input window | 168 hourly steps (7 days) |
| Input features | 9 |
| Output | One next-step prediction |
| Optimizer | Adam |
| Loss function | Mean Squared Error (MSE) |
| Training approach | Sequential train/validation/test split |

**Features:** `solar_gen`, `temperature`, `humidity`, `pressure`, `hour`, `month`, `cloud_cover`, `irradiance`, `panel_efficiency`.

Both training scripts use 10 epochs and a batch size of 32. The current preprocessing fits the scaler before the chronological split, which may introduce data leakage. A future improvement is to fit the scaler on training data only and save it for consistent inference.

## 🔀 Fuzzy Logic Engine

The fuzzy system accepts five inputs:

| Input | Range |
|---|---:|
| `energy_demand` | 0–100 |
| `solar_gen` | 0–100 |
| `battery_level` | 0–100 |
| `grid_price` | 0–100 |
| `time_of_day` | 0–24 |

It produces three outputs:

| Output | Range |
|---|---:|
| `appliance_reduction` | 0–100 |
| `battery_action` | -100 to +100 |
| `grid_interaction` | -100 to +100 |

These are normalized decision-system values, not calibrated physical measurements. Refer to `fuzzy_system.py` for the active rules and implementation.

## 📊 Dashboard Tabs

| Tab | Purpose |
|---|---|
| Overview | Summary metrics and energy information |
| Predictions | Consumption and solar model demonstrations |
| AI Decisions | Fuzzy-system outputs and recommendations |
| Analytics | Baseline-versus-optimized demand and scenario costs |

The dashboard includes interactive controls for exploring simulated energy conditions.

## ⚙️ Installation and Usage

### Requirements

- Python 3.12 (development environment)
- pip
- Project dependencies listed below

### Run on macOS

Open Terminal and run:

```bash
cd ~/Desktop/Smart-Home-energy-consumption
source .venv/bin/activate
```

If dependencies are not installed in the virtual environment:

```bash
python -m pip install streamlit pandas numpy tensorflow plotly scikit-fuzzy networkx scikit-learn matplotlib
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

Open the local URL displayed in Terminal, usually `http://localhost:8501`.

Stop the dashboard with **Control + C** in Terminal.

If you do not already have a virtual environment, create one first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## ⚡ Energy Analytics Simulator

The `energy_optimizer.py` module provides three main functions:

- `load_energy_data(...)` — loads the consumption and solar datasets, parses timestamps, and aligns the data.
- `simulate_optimization(data, flexible_percent=15)` — simulates shifting a configurable share of demand toward same-day hours with solar surplus.
- `summarize_scenarios(data, tariff_per_unit=8.0)` — compares baseline and simulated grid demand and estimates illustrative costs using an assumed tariff.

The optional nighttime solar correction sets simulated solar generation to zero before 06:00 and from 20:00 onward. This is an explicit simulation assumption, not a verified correction to the source data.

## 📈 Data and Validation

The datasets contain 8,760 hourly records spanning 1 January through 31 December 2025.

Development checks performed include:

- Matching timestamps across both datasets.
- A sample simulation preserving total demand while shifting its timing.
- Checks for finite simulation outputs.
- A passing zero-flexibility scenario test.
- Successful model loading and finite outputs for correctly shaped smoke-test inputs.

**These checks do not establish predictive accuracy.** Model smoke tests only verify basic execution. Forecast performance must be evaluated on held-out chronological data using metrics such as MAE and RMSE and compared against a baseline.

## ⚠️ Assumptions and Limitations

- The physical units of the datasets have not been verified; values should not be presented as confirmed kWh.
- Estimated costs are illustrative and depend on the assumed tariff. They are not verified electricity-bill savings.
- Negative readings are clipped for simulation calculations; the source CSV files remain unchanged.
- Nighttime solar correction is optional and assumption-based.
- Demand shifting is a heuristic, not a validated optimization solver.
- The LSTM forecasts are not yet integrated directly into the analytics simulator.
- The project is not designed for direct control of real electrical equipment.

## 🚀 Future Improvements

1. Evaluate forecasts using MAE and RMSE on a held-out chronological test set.
2. Fit scalers on training data only and persist them for inference.
3. Integrate LSTM predictions into the optimizer after validating the data pipeline.
4. Expand automated tests for fuzzy-system boundaries and invalid inputs.
5. Verify the dataset's physical units and investigate anomalous readings.
6. Add a pinned `requirements.txt` and reproducible setup instructions.

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- LSTM neural networks
- Streamlit
- Plotly
- scikit-fuzzy
- scikit-learn
- Pandas and NumPy

## 👩‍💻 Project Information

**Project:** EcoMind AI  
**Project branding:** Sara  
**Academic context:** Intelligent Systems-II mini-project  
**Focus:** LSTM forecasting, fuzzy logic, and home-energy scenario analysis
