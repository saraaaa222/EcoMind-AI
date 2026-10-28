import numpy as np
import skfuzzy.control as ctrl
from dashboard import setup_fuzzy_system

system = setup_fuzzy_system()
simulation = ctrl.ControlSystemSimulation(system)

scenarios = [
    {
        "name": "High demand, low solar, low battery",
        "energy_demand": 80,
        "solar_gen": 20,
        "battery_level": 25,
        "grid_price": 70,
        "time_of_day": 19,
    },
    {
        "name": "Low demand, high solar",
        "energy_demand": 20,
        "solar_gen": 85,
        "battery_level": 60,
        "grid_price": 20,
        "time_of_day": 12,
    },
    {
        "name": "Medium demand, moderate solar",
        "energy_demand": 50,
        "solar_gen": 50,
        "battery_level": 60,
        "grid_price": 50,
        "time_of_day": 15,
    },
]

for scenario in scenarios:
    print(f"\n--- {scenario['name']} ---")
    simulation = ctrl.ControlSystemSimulation(system)

    for key in (
        "energy_demand",
        "solar_gen",
        "battery_level",
        "grid_price",
    ):
        simulation.input[key] = scenario[key]

    try:
        simulation.compute()
        for output_name, value in simulation.output.items():
            print(f"{output_name}: {value:.2f}")
    except Exception as error:
        print(f"TEST FAILED: {error}")
