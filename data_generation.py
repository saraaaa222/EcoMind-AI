import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create fake data for energy consumption
start_date = datetime(2025, 1, 1)
hours = 8760  # One year of hourly data
dates = [start_date + timedelta(hours=i) for i in range(hours)]

# Fake consumption: higher in evenings, weekends
consumption = 5 + np.sin(np.arange(hours) / 24 * 2 * np.pi) * 3 + np.random.normal(0, 1, hours)
temperature = 20 + np.sin(np.arange(hours) / 24 * 2 * np.pi) * 10 + np.random.normal(0, 2, hours)
hour = [d.hour for d in dates]
day_of_week = [d.weekday() for d in dates]
month = [d.month for d in dates]
occupancy = np.random.choice([0, 1], hours, p=[0.3, 0.7])  # 70% occupied
weather = np.random.uniform(0, 1, hours)  # 0 bad, 1 good
previous_day_cons = np.roll(consumption, 24)  # Shift by 24 hours
weekend = [1 if dow >= 5 else 0 for dow in day_of_week]

data_cons = pd.DataFrame({
    'timestamp': dates,
    'consumption': consumption,
    'temperature': temperature,
    'hour': hour,
    'day_of_week': day_of_week,
    'month': month,
    'occupancy': occupancy,
    'weather': weather,
    'previous_day_cons': previous_day_cons,
    'weekend': weekend
})
data_cons.to_csv('consumption_data.csv', index=False)

# Fake solar data
solar_gen = np.maximum(0, np.sin(np.arange(hours) / 24 * np.pi) * 10 + np.random.normal(0, 2, hours))  # Peak at noon
humidity = np.random.uniform(30, 90, hours)
pressure = np.random.uniform(900, 1100, hours)
cloud_cover = np.random.uniform(0, 1, hours)
irradiance = (1 - cloud_cover) * 1000
panel_efficiency = 0.2 + np.random.normal(0, 0.01, hours)

data_solar = pd.DataFrame({
    'timestamp': dates,
    'solar_gen': solar_gen,
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure,
    'hour': hour,
    'month': month,
    'cloud_cover': cloud_cover,
    'irradiance': irradiance,
    'panel_efficiency': panel_efficiency
})
data_solar.to_csv('solar_data.csv', index=False)

# Note for antigravity: In space, no dust on panels, so efficiency +10%
data_solar['panel_efficiency'] += 0.1
data_solar.to_csv('solar_data.csv', index=False)  # Overwrite with adjustment

print("Fake data created: consumption_data.csv and solar_data.csv")