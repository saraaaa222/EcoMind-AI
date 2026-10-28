import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define universes (scales)
energy_demand = ctrl.Antecedent(np.arange(0, 101, 1), 'energy_demand')
solar_gen = ctrl.Antecedent(np.arange(0, 101, 1), 'solar_gen')
battery_level = ctrl.Antecedent(np.arange(0, 101, 1), 'battery_level')
grid_price = ctrl.Antecedent(np.arange(0, 101, 1), 'grid_price')
time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')

appliance_reduction = ctrl.Consequent(np.arange(0, 101, 1), 'appliance_reduction')
battery_action = ctrl.Consequent(np.arange(-100, 101, 1), 'battery_action')
grid_interaction = ctrl.Consequent(np.arange(-100, 101, 1), 'grid_interaction')

# Membership functions
# Energy Demand
energy_demand['low'] = fuzz.trimf(energy_demand.universe, [0, 0, 30])
energy_demand['medium'] = fuzz.trimf(energy_demand.universe, [20, 50, 80])
energy_demand['high'] = fuzz.trimf(energy_demand.universe, [70, 100, 100])

# Solar Gen
solar_gen['poor'] = fuzz.trimf(solar_gen.universe, [0, 0, 30])
solar_gen['moderate'] = fuzz.trimf(solar_gen.universe, [20, 50, 80])
solar_gen['excellent'] = fuzz.trimf(solar_gen.universe, [70, 100, 100])

# Battery Level
battery_level['critical'] = fuzz.trimf(battery_level.universe, [0, 0, 20])
battery_level['low'] = fuzz.trimf(battery_level.universe, [10, 30, 50])
battery_level['medium'] = fuzz.trimf(battery_level.universe, [40, 60, 80])
battery_level['high'] = fuzz.trimf(battery_level.universe, [70, 100, 100])

# Grid Price
grid_price['cheap'] = fuzz.trimf(grid_price.universe, [0, 0, 30])
grid_price['normal'] = fuzz.trimf(grid_price.universe, [20, 50, 80])
grid_price['expensive'] = fuzz.trimf(grid_price.universe, [70, 100, 100])

# Time of Day
time_of_day['night1'] = fuzz.trimf(time_of_day.universe, [0, 3, 6])
time_of_day['night2'] = fuzz.trimf(time_of_day.universe, [22, 23, 24])
time_of_day['night'] = np.maximum(time_of_day['night1'].mf, time_of_day['night2'].mf)  # Union for wrap-around
time_of_day['morning'] = fuzz.trapmf(time_of_day.universe, [6, 8, 10, 12])
time_of_day['afternoon'] = fuzz.trapmf(time_of_day.universe, [12, 14, 16, 18])
time_of_day['evening'] = fuzz.trimf(time_of_day.universe, [18, 20, 22])

# Outputs
appliance_reduction['none'] = fuzz.trimf(appliance_reduction.universe, [0, 0, 20])
appliance_reduction['slight'] = fuzz.trimf(appliance_reduction.universe, [10, 30, 50])
appliance_reduction['moderate'] = fuzz.trimf(appliance_reduction.universe, [40, 60, 80])
appliance_reduction['aggressive'] = fuzz.trimf(appliance_reduction.universe, [70, 100, 100])

battery_action['discharge'] = fuzz.trimf(battery_action.universe, [-100, -100, -50])
battery_action['maintain'] = fuzz.trimf(battery_action.universe, [-30, 0, 30])
battery_action['charge'] = fuzz.trimf(battery_action.universe, [50, 100, 100])

grid_interaction['sell'] = fuzz.trimf(grid_interaction.universe, [50, 100, 100])
grid_interaction['neutral'] = fuzz.trimf(grid_interaction.universe, [-30, 0, 30])
grid_interaction['buy'] = fuzz.trimf(grid_interaction.universe, [-100, -100, -50])

# 20+ Rules (covering scenarios)
rules = []

# Peak Demand Management (high demand cases)
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['poor'] & battery_level['low'] & grid_price['expensive'] & time_of_day['evening'], 
                       (appliance_reduction['aggressive'], battery_action['discharge'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['moderate'] & battery_level['medium'] & grid_price['normal'] & time_of_day['night'], 
                       (appliance_reduction['moderate'], battery_action['discharge'], grid_interaction['buy'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['excellent'] & battery_level['high'] & grid_price['cheap'] & time_of_day['morning'], 
                       (appliance_reduction['slight'], battery_action['maintain'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['poor'] & battery_level['critical'] & grid_price['expensive'] & time_of_day['afternoon'], 
                       (appliance_reduction['aggressive'], battery_action['maintain'], grid_interaction['buy'])))

# Solar Optimization (good solar cases)
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['excellent'] & battery_level['low'] & grid_price['expensive'] & time_of_day['afternoon'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['sell'])))
rules.append(ctrl.Rule(energy_demand['medium'] & solar_gen['excellent'] & battery_level['medium'] & grid_price['normal'] & time_of_day['morning'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['moderate'] & battery_level['high'] & grid_price['cheap'] & time_of_day['afternoon'], 
                       (appliance_reduction['none'], battery_action['maintain'], grid_interaction['sell'])))

# Battery Management
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['poor'] & battery_level['critical'] & grid_price['cheap'] & time_of_day['night'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['buy'])))
rules.append(ctrl.Rule(energy_demand['medium'] & solar_gen['moderate'] & battery_level['low'] & grid_price['expensive'] & time_of_day['evening'], 
                       (appliance_reduction['slight'], battery_action['maintain'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['excellent'] & battery_level['critical'] & grid_price['normal'] & time_of_day['afternoon'], 
                       (appliance_reduction['moderate'], battery_action['charge'], grid_interaction['sell'])))

# Grid Decisions
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['excellent'] & battery_level['high'] & grid_price['expensive'] & time_of_day['morning'], 
                       (appliance_reduction['none'], battery_action['maintain'], grid_interaction['sell'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['poor'] & battery_level['medium'] & grid_price['cheap'] & time_of_day['night'], 
                       (appliance_reduction['moderate'], battery_action['discharge'], grid_interaction['buy'])))

# Emergency Scenarios (e.g., critical battery, high demand)
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['poor'] & battery_level['critical'] & grid_price['normal'] & time_of_day['evening'], 
                       (appliance_reduction['aggressive'], battery_action['maintain'], grid_interaction['buy'])))
rules.append(ctrl.Rule(energy_demand['medium'] & solar_gen['poor'] & battery_level['critical'] & grid_price['expensive'] & time_of_day['night'], 
                       (appliance_reduction['moderate'], battery_action['maintain'], grid_interaction['neutral'])))

# More to reach 20: Variations
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['moderate'] & battery_level['medium'] & grid_price['cheap'] & time_of_day['morning'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['buy'])))
rules.append(ctrl.Rule(energy_demand['medium'] & solar_gen['excellent'] & battery_level['high'] & grid_price['expensive'] & time_of_day['afternoon'], 
                       (appliance_reduction['slight'], battery_action['charge'], grid_interaction['sell'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['moderate'] & battery_level['low'] & grid_price['normal'] & time_of_day['evening'], 
                       (appliance_reduction['moderate'], battery_action['discharge'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['poor'] & battery_level['high'] & grid_price['cheap'] & time_of_day['night'], 
                       (appliance_reduction['none'], battery_action['maintain'], grid_interaction['buy'])))
rules.append(ctrl.Rule(energy_demand['medium'] & solar_gen['poor'] & battery_level['medium'] & grid_price['expensive'] & time_of_day['morning'], 
                       (appliance_reduction['slight'], battery_action['discharge'], grid_interaction['neutral'])))
rules.append(ctrl.Rule(energy_demand['high'] & solar_gen['excellent'] & battery_level['high'] & grid_price['normal'] & time_of_day['afternoon'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['sell'])))
rules.append(ctrl.Rule(energy_demand['low'] & solar_gen['excellent'] & battery_level['critical'] & grid_price['expensive'] & time_of_day['evening'], 
                       (appliance_reduction['none'], battery_action['charge'], grid_interaction['neutral'])))

# Control system
energy_ctrl = ctrl.ControlSystem(rules)
energy_sim = ctrl.ControlSystemSimulation(energy_ctrl)

# Save? Not needed, we use in dashboard

# Test example
energy_sim.input['energy_demand'] = 80
energy_sim.input['solar_gen'] = 20
energy_sim.input['battery_level'] = 40
energy_sim.input['grid_price'] = 70
energy_sim.input['time_of_day'] = 15

energy_sim.compute()

print("Appliance Reduction:", energy_sim.output['appliance_reduction'])
print("Battery Action:", energy_sim.output['battery_action'])
print("Grid Interaction:", energy_sim.output['grid_interaction'])

# Antigravity note: In space, add rule if needed for constant light, but here solar assumes Earth-like.

print("Fuzzy system ready.")