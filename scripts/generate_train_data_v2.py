
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ===================
# CONFIGURATION (edit)
# ===================
# Sampling rate: 0.02 s ~ 50 Hz (can change to 0.5, 0.01, etc.)
SAMPLING_RATE_SECONDS = 0.02

# List of random seeds to generate multiple variations
SEEDS = [7, 21, 42]  # add/remove seeds as you like

# Duration of each motion segment in SECONDS
# Total duration should be sum of all segments below
SEGMENTS_SPEC = [
    ("walk_1", 50),   # 50s
    ("turn_1", 10),   # 10s
    ("rest",   80),   # 80s
    ("walk_2", 50),   # 50s
    ("jump_2", 10),   # 10s
]
# With 50+10+80+50+10 = 200 seconds; at 50 Hz -> 10,000 rows

# Base noise (std). Each run will jitter these around ±20% by seed.
ACCEL_NOISE_STD_BASE = 0.05
GYRO_NOISE_STD_BASE  = 0.005

# Drift ranges (per second) and constant bias ranges (both randomized per axis)
# Units roughly in m/s^2 for accel, rad/s for gyro
ACCEL_BIAS_RANGE  = (-0.15, 0.15)   # static offset
ACCEL_DRIFT_RANGE = (-0.002, 0.002) # linear drift per second
GYRO_BIAS_RANGE   = (-0.02, 0.02)
GYRO_DRIFT_RANGE  = (-0.0005, 0.0005)

# Output directory
OUT_DIR = Path("data/raw")

# Timezone UTC+7 as in previous dataset
TZ_OFFSET = timezone(timedelta(hours=7))

# ==============
# Helper methods
# ==============
def seconds_to_rows(seconds, dt):
    return int(round(seconds / dt))

def build_timestamp_series(start_time, n_rows, dt):
    return [start_time + timedelta(seconds=i * dt) for i in range(n_rows)]

def apply_bias_and_drift(df, dt, accel_axes=("Accel_x","Accel_y","Accel_z"), gyro_axes=("Gyro_x","Gyro_y","Gyro_z"), rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n = len(df)
    t_seconds = np.arange(n) * dt

    # Random bias + drift per axis
    for axis in accel_axes:
        bias  = rng.uniform(*ACCEL_BIAS_RANGE)
        drift = rng.uniform(*ACCEL_DRIFT_RANGE)
        df[axis] += bias + drift * t_seconds

    for axis in gyro_axes:
        bias  = rng.uniform(*GYRO_BIAS_RANGE)
        drift = rng.uniform(*GYRO_DRIFT_RANGE)
        df[axis] += bias + drift * t_seconds

def generate_segment(df, start_idx, end_idx, label, dt, rng, accel_params, gyro_params):
    """Fill df[start_idx:end_idx] with motion for the given label."""
    idx = slice(start_idx, end_idx)
    n = end_idx - start_idx
    t_local = np.arange(n) * dt  # local time within the segment

    # Unpack randomized parameters
    acc_amp_y, acc_amp_z, walk_freq_hz, gyro_amp_z = accel_params
    turn_gyro_max, turn_ax_min, turn_ax_max = gyro_params

    if label.startswith("walk"):
        # Smooth walking oscillations
        # y-axis sway and z-axis slight bounce (around gravity)
        df.loc[idx, "Accel_y"] += acc_amp_y * np.sin(2*np.pi*walk_freq_hz * t_local + rng.uniform(0, 2*np.pi))
        df.loc[idx, "Accel_z"] += 9.8 + acc_amp_z * np.cos(2*np.pi*walk_freq_hz * t_local + rng.uniform(0, 2*np.pi))
        df.loc[idx, "Gyro_z"]  += gyro_amp_z * np.sin(2*np.pi*(walk_freq_hz/2) * t_local + rng.uniform(0, 2*np.pi))

    elif label == "turn_1":
        # Sharp yaw turn: ramp Gyro_z; slight lateral accel_x change
        df.loc[idx, "Gyro_z"] = np.linspace(0.0, turn_gyro_max, n)
        df.loc[idx, "Accel_x"] = np.linspace(turn_ax_min, turn_ax_max, n)
        df.loc[idx, "Accel_z"] += 9.8  # keep gravity baseline visible

    elif label == "rest":
        # Stillness: near-baseline around gravity
        df.loc[idx, "Accel_z"] += 9.8

    elif label == "jump_2":
        # Quick jump: spike in Accel_z + chaotic gyro x/y correlating with spike
        half = n // 2
        spike = np.concatenate([np.linspace(0, 5, half), np.linspace(5, 0, n - half)])
        df.loc[idx, "Accel_z"] += 9.8 + spike
        chaos = (spike / 5.0 + 0.1)
        df.loc[idx, "Gyro_x"] += 2.0 * rng.standard_normal(n) * chaos
        df.loc[idx, "Gyro_y"] += 2.0 * rng.standard_normal(n) * chaos

    else:
        # Default: keep gravity baseline to avoid unrealistic zero-g
        df.loc[idx, "Accel_z"] += 9.8

    # Set label
    df.loc[idx, "ActivityLabel"] = label

def generate_dataset(seed=42, dt=SAMPLING_RATE_SECONDS, start_time=None):
    rng = np.random.default_rng(seed)

    # Randomize noise around base (±20% jitter)
    accel_noise = ACCEL_NOISE_STD_BASE * rng.uniform(0.8, 1.2)
    gyro_noise  = GYRO_NOISE_STD_BASE  * rng.uniform(0.8, 1.2)

    # Compute rows for each segment
    seg_rows = [seconds_to_rows(sec, dt) for _, sec in SEGMENTS_SPEC]
    total_rows = sum(seg_rows)

    # Initialize frame
    df = pd.DataFrame(index=range(total_rows), columns=[
        "Timestamp", "Accel_x", "Accel_y", "Accel_z", "Gyro_x", "Gyro_y", "Gyro_z", "ActivityLabel"
    ], dtype=float)
    df[["Accel_x","Accel_y","Accel_z","Gyro_x","Gyro_y","Gyro_z"]] = 0.0
    df["ActivityLabel"] = "rest"  # default

    # Per-run randomized motion parameters
    acc_amp_y     = rng.uniform(1.0, 2.0)     # sway (m/s^2)
    acc_amp_z     = rng.uniform(0.3, 0.7)     # vertical bounce (m/s^2)
    walk_freq_hz  = rng.uniform(0.8, 1.8)     # step frequency (Hz)
    gyro_amp_z    = rng.uniform(0.8, 1.6)     # yaw oscillation (rad/s)

    turn_gyro_max = rng.uniform(2.5, 4.5)     # max yaw rate during turn
    turn_ax_min   = rng.uniform(0.8, 1.5)
    turn_ax_max   = rng.uniform(2.0, 3.5)

    # Fill segments
    cursor = 0
    for (label, secs), rows in zip(SEGMENTS_SPEC, seg_rows):
        generate_segment(
            df, cursor, cursor+rows, label, dt, rng,
            accel_params=(acc_amp_y, acc_amp_z, walk_freq_hz, gyro_amp_z),
            gyro_params=(turn_gyro_max, turn_ax_min, turn_ax_max)
        )
        cursor += rows

    # Add measurement noise
    for col in ["Accel_x","Accel_y","Accel_z"]:
        df[col] += rng.normal(0, accel_noise, len(df))
    for col in ["Gyro_x","Gyro_y","Gyro_z"]:
        df[col] += rng.normal(0, gyro_noise, len(df))

    # Apply slow bias and linear drift
    apply_bias_and_drift(df, dt, rng=rng)

    # Timestamp
    if start_time is None:
        start_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=TZ_OFFSET)
    df["Timestamp"] = build_timestamp_series(start_time, len(df), dt)

    # Final ordering & formatting
    cols = ["Timestamp","Accel_x","Accel_y","Accel_z","Gyro_x","Gyro_y","Gyro_z","ActivityLabel"]
    df = df[cols]
    df["Timestamp"] = df["Timestamp"].apply(lambda x: x.isoformat(timespec="milliseconds"))

    return df

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=TZ_OFFSET)
    stamp = now.strftime("%Y%m%d_%H%M%S")

    for seed in SEEDS:
        df = generate_dataset(seed=seed, dt=SAMPLING_RATE_SECONDS)
        fname = OUT_DIR / f"synthetic_training_data_{stamp}_seed{seed}_sr{int(round(1/SAMPLING_RATE_SECONDS))}Hz.csv"
        df.to_csv(fname, index=False)
        print(f"Saved: {fname}  (rows={len(df)})")

if __name__ == "__main__":
    main()
