import numpy as np
import pandas as pd


def generate_dataset(n=250, version=1, seed=42):
    """version 1 = normal evenings, version 2 = college festival (higher demand)."""
    rng = np.random.default_rng(seed + version)
    festival = version == 2

    charged = np.clip(rng.normal(34, 6, n).round(), 10, 55).astype(int)
    finishing = rng.integers(0, 16, n)
    bookings = rng.poisson(12 if festival else 11, n)
    walkin = np.clip(rng.normal(26 if festival else 24, 6, n).round(), 5, None).astype(int)
    rain = np.clip(rng.exponential(5, n), 0, 40).round(1)
    rain[rng.random(n) < 0.5] = 0.0
    weekend = rng.integers(0, 2, n)
    event = (rng.random(n) < (0.4 if festival else 0.15)).astype(int)

    # demand multiplier: rain, weekend, nearby event, festival boost
    mult = 1 + 0.015 * rain + 0.15 * weekend + 0.30 * event + (0.05 if festival else 0.0)

    # simulated customer arrivals
    arrivals = rng.binomial(bookings, 0.9) + rng.poisson(walkin * mult)
    # simulated actual supply at 7 PM (90% of charging batteries finish on time)
    supply = charged + rng.binomial(finishing, 0.9)

    df = pd.DataFrame({
        "charged_batteries_5pm": charged,
        "batteries_finishing_by_7pm": finishing,
        "advance_bookings": bookings,
        "historical_walkin_demand": walkin,
        "rainfall_forecast_mm": rain,
        "is_weekend": weekend,
        "nearby_event": event,
        "simulated_arrivals": arrivals,
    })
    df["battery_shortage"] = (arrivals > supply).astype(int)
    df["dataset_version"] = version
    return df


if __name__ == "__main__":
    for v in (1, 2):
        d = generate_dataset(version=v)
        print(f"v{v}: rows={len(d)}, shortage rate={d.battery_shortage.mean():.2%}")
