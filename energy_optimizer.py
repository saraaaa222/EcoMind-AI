
import pandas as pd


def load_energy_data(
    consumption_path="consumption_data.csv",
    solar_path="solar_data.csv",
    correct_nighttime_solar=True,
):
    """Load and align the original hourly datasets without modifying the CSVs."""
    consumption = pd.read_csv(consumption_path)
    solar = pd.read_csv(solar_path)

    required_consumption = {"timestamp", "consumption"}
    required_solar = {"timestamp", "solar_gen"}

    if not required_consumption.issubset(consumption.columns):
        raise ValueError("Consumption CSV must contain timestamp and consumption.")
    if not required_solar.issubset(solar.columns):
        raise ValueError("Solar CSV must contain timestamp and solar_gen.")

    consumption["timestamp"] = pd.to_datetime(
        consumption["timestamp"], errors="coerce"
    )
    solar["timestamp"] = pd.to_datetime(solar["timestamp"], errors="coerce")

    consumption = consumption.dropna(subset=["timestamp", "consumption"])
    solar = solar.dropna(subset=["timestamp", "solar_gen"])

    data = consumption[["timestamp", "consumption"]].merge(
        solar[["timestamp", "solar_gen"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError("No matching timestamps found between the two CSV files.")

    # Keep source data intact; clip negative readings only in this simulation.
    data["demand"] = data["consumption"].clip(lower=0)
    data["solar"] = data["solar_gen"].clip(lower=0)
    data["hour"] = data["timestamp"].dt.hour

    if correct_nighttime_solar:
        night = (data["hour"] < 6) | (data["hour"] >= 20)
        data.loc[night, "solar"] = 0.0

    data["baseline_grid"] = (data["demand"] - data["solar"]).clip(lower=0)
    return data.sort_values("timestamp").reset_index(drop=True)


def simulate_optimization(data, flexible_percent=15):
    """
    Simulate shifting a configurable share of demand into same-day solar-surplus
    hours. This is a heuristic scenario, not a learned prediction or guarantee.
    """
    if not 0 <= flexible_percent <= 30:
        raise ValueError("flexible_percent must be between 0 and 30.")

    result = data.copy()
    result["optimized_demand"] = result["demand"].astype(float)

    if flexible_percent == 0:
        result["optimized_grid"] = result["baseline_grid"]
        return result

    result["date"] = result["timestamp"].dt.date

    for _, indices_group in result.groupby("date").groups.items():
        indices = list(indices_group)

        donors = sorted(
            [i for i in indices if result.at[i, "demand"] > result.at[i, "solar"]],
            key=lambda i: result.at[i, "demand"] - result.at[i, "solar"],
            reverse=True,
        )
        recipients = sorted(
            [i for i in indices if result.at[i, "solar"] > result.at[i, "demand"]],
            key=lambda i: result.at[i, "solar"] - result.at[i, "demand"],
            reverse=True,
        )

        for donor in donors:
            movable = min(
                result.at[donor, "demand"] * flexible_percent / 100,
                max(
                    result.at[donor, "optimized_demand"] - result.at[donor, "solar"],
                    0,
                ),
            )

            for recipient in recipients:
                if movable <= 1e-9:
                    break

                surplus = max(
                    result.at[recipient, "solar"]
                    - result.at[recipient, "optimized_demand"],
                    0,
                )
                transfer = min(movable, surplus)

                if transfer > 0:
                    result.at[donor, "optimized_demand"] -= transfer
                    result.at[recipient, "optimized_demand"] += transfer
                    movable -= transfer

    result["optimized_grid"] = (
        result["optimized_demand"] - result["solar"]
    ).clip(lower=0)

    return result.drop(columns=["date"], errors="ignore")


def summarize_scenarios(data, tariff_per_unit=8.0):
    """Return clearly labelled scenario metrics; units must be independently verified."""
    baseline_grid = float(data["baseline_grid"].sum())
    optimized_grid = float(data["optimized_grid"].sum())
    reduction = baseline_grid - optimized_grid

    return {
        "matched_hours": int(len(data)),
        "baseline_demand_units": float(data["demand"].sum()),
        "solar_generation_units": float(data["solar"].sum()),
        "baseline_grid_units": baseline_grid,
        "optimized_grid_units": optimized_grid,
        "grid_reduction_units": reduction,
        "grid_reduction_percent": (
            100 * reduction / baseline_grid if baseline_grid else 0.0
        ),
        "illustrative_baseline_cost": baseline_grid * tariff_per_unit,
        "illustrative_optimized_cost": optimized_grid * tariff_per_unit,
        "illustrative_cost_difference": reduction * tariff_per_unit,
        "tariff_per_assumed_unit": float(tariff_per_unit),
    }
