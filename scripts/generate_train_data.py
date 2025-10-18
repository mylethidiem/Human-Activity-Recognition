import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- Configuration ---
NUM_ROWS = 10000              # Training file size (10,000 data points)
SAMPLING_RATE_SECONDS = 0.5   # Time step (0.5 seconds)
ACCEL_NOISE_STD = 0.05
GYRO_NOISE_STD = 0.005

# Define the fixed UTC+7 timezone offset based on the requested format
TZ_OFFSET = timezone(timedelta(hours=7))

# --- 1. Data Initialization and Timestamp Generation ---
# Apply the timezone offset to the start_time
start_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=TZ_OFFSET) # Training file start date
timestamps = [start_time + timedelta(seconds=i * SAMPLING_RATE_SECONDS) for i in range(NUM_ROWS)]
data = pd.DataFrame(index=range(NUM_ROWS))
t = np.linspace(0, NUM_ROWS * SAMPLING_RATE_SECONDS, NUM_ROWS)

# Define motion segments:
# Reduced rest to 4000 rows (40%) to ensure median falls in active region (60%).
SEGMENTS = {
    # Block 1: Initial Activity (0 - 3000 rows)
    'walk_1': (0, 2500),     # Initial Steady Walking (2500 rows)
    'turn_1': (2500, 3000),  # Initial Sharp Turn (500 rows)

    # Block 2: Long Rest/Gap (3000 - 7000 rows) - Creates the flat line in the plot
    'long_rest': (3000, 7000), # Resting/Stillness (4000 rows)

    # Block 3: Final Activity (7000 - 10000 rows)
    'walk_2': (7000, 9500),  # Second Walking Phase (2500 rows)
    'jump_2': (9500, NUM_ROWS),  # Quick Jump/Impact (500 rows)
}

# Base signal setup: Accel Z starts with gravity (9.8 m/s^2)
data['Accel_x'] = 0.0
data['Accel_y'] = 0.0
data['Accel_z'] = 9.8
data['Gyro_x'] = 0.0
data['Gyro_y'] = 0.0
data['Gyro_z'] = 0.0

# --- 2. Motion Profile Simulation ---

# Phase A: Steady Walking 1
idx_start, idx_end = SEGMENTS['walk_1']
t_walk = t[idx_start:idx_end]
data.loc[idx_start:idx_end-1, 'Accel_y'] += 1.5 * np.sin(4 * np.pi * t_walk / 10)
data.loc[idx_start:idx_end-1, 'Accel_z'] += 0.5 * np.cos(4 * np.pi * t_walk / 10)
# CRITICAL FIX: Gyro Z amplitude increased to 1.26 to target median 0.36851
data.loc[idx_start:idx_end-1, 'Gyro_x'] += 0.0
data.loc[idx_start:idx_end-1, 'Gyro_y'] += 0.0
data.loc[idx_start:idx_end-1, 'Gyro_z'] += 1.26 * np.sin(2 * np.pi * t_walk / 10)

# Phase B: Sharp Turn 1
idx_start, idx_end = SEGMENTS['turn_1']
data.loc[idx_start:idx_end-1, 'Gyro_z'] = np.linspace(0.0, 4.0, idx_end - idx_start)
data.loc[idx_start:idx_end-1, 'Accel_x'] = np.linspace(1.0, 3.0, idx_end - idx_start)

# Phase C: Long Rest (3000-7000) - Data is at the base signal (Accel_z=9.8, others=0.0)

# Phase D: Steady Walking 2 (Mirroring the first walk, but later in time)
idx_start, idx_end = SEGMENTS['walk_2']
t_walk = t[idx_start:idx_end]
data.loc[idx_start:idx_end-1, 'Accel_y'] += 1.5 * np.sin(4 * np.pi * t_walk / 10)
data.loc[idx_start:idx_end-1, 'Accel_z'] += 0.5 * np.cos(4 * np.pi * t_walk / 10)
# CRITICAL FIX: Gyro Z amplitude increased to 1.26 to target median 0.36851
data.loc[idx_start:idx_end-1, 'Gyro_x'] += 0.0
data.loc[idx_start:idx_end-1, 'Gyro_y'] += 0.0
data.loc[idx_start:idx_end-1, 'Gyro_z'] += 1.26 * np.sin(2 * np.pi * t_walk / 10)

# Phase E: Jump 2 (Quick vertical spike and uncontrolled rotation)
idx_start, idx_end = SEGMENTS['jump_2']
segment_length = idx_end - idx_start # 500 rows
spike_amplitude = np.concatenate([np.linspace(0, 5, segment_length // 2), np.linspace(5, 0, segment_length // 2)])
data.loc[idx_start:idx_end-1, 'Accel_z'] += spike_amplitude
# Gyro X/Y chaos factor remains at 2.0
data.loc[idx_start:idx_end-1, 'Gyro_x'] += 2.0 * np.random.randn(segment_length) * (spike_amplitude/5 + 0.1)
data.loc[idx_start:idx_end-1, 'Gyro_y'] += 2.0 * np.random.randn(segment_length) * (spike_amplitude/5 + 0.1)


# --- 3. Add Measurement Noise ---
data['Accel_x'] += np.random.normal(0, ACCEL_NOISE_STD, NUM_ROWS)
data['Accel_y'] += np.random.normal(0, ACCEL_NOISE_STD, NUM_ROWS)
data['Accel_z'] += np.random.normal(0, ACCEL_NOISE_STD, NUM_ROWS)
data['Gyro_x'] += np.random.normal(0, GYRO_NOISE_STD, NUM_ROWS)
data['Gyro_y'] += np.random.normal(0, GYRO_NOISE_STD, NUM_ROWS)
data['Gyro_z'] += np.random.normal(0, GYRO_NOISE_STD, NUM_ROWS)

# --------------------------------------------------------------------------
# --- 4. Median Calculation (Temporary) ---
# Calculate the Euclidean norms as described in your analysis code
data['acc_norm'] = np.linalg.norm(data[['Accel_x','Accel_y','Accel_z']].values, axis=1)
data['gyro_norm'] = np.linalg.norm(data[['Gyro_x','Gyro_y','Gyro_z']].values, axis=1)

# Compute the median values
acc_med_calculated = data['acc_norm'].median()
gyro_med_calculated = data['gyro_norm'].median()

# Remove temporary columns before saving to keep the CSV clean
data = data.drop(columns=['acc_norm', 'gyro_norm'])
# --------------------------------------------------------------------------


# --- 5. Final Formatting and Saving ---

data['Timestamp'] = timestamps # Changed column name back to 'Timestamp'
columns_order = ['Timestamp', 'Accel_x', 'Accel_y', 'Accel_z', 'Gyro_x', 'Gyro_y', 'Gyro_z'] # Updated order
final_df = data[columns_order]

# Format Timestamp using isoformat()
final_df['Timestamp'] = final_df['Timestamp'].apply(lambda x: x.isoformat(timespec='milliseconds'))

# Generate timestamped filename for the training set
now = datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M%S")
file_name = f'data/raw/synthetic_training_data_{timestamp_str}.csv'

# Save the full 10,000 rows
final_df.to_csv(file_name, index=False)

print(f"✅ Training data generation complete. The full {len(final_df)} rows were saved to the file:\n   '{file_name}'")
print("\n--- Calculated Statistical Medians ---")
# The median should now be in the target range due to the minimum activity floor.
print(f"Accel Norm Median (acc_med): {acc_med_calculated}")
print(f"Gyro Norm Median (gyro_med): {gyro_med_calculated}")
print("\n--- CSV Data Range Confirmation (Max/Min Values) ---")
print("Accel Max: ", final_df[['Accel_x', 'Accel_y', 'Accel_z']].max().max())
print("Accel Min: ", final_df[['Accel_x', 'Accel_y', 'Accel_z']].min().min())
print("Gyro Max: ", final_df[['Gyro_x', 'Gyro_y', 'Gyro_z']].max().max())
print("Gyro Min: ", final_df[['Gyro_x', 'Gyro_y', 'Gyro_z']].min().min())
