# 🧠 Human Activity Recognition Using EDA Sensor Data

This project focuses on **recognizing human activities** and **relative movement patterns** based on multi-sensor data (Accelerometer, Gyroscope, and EDA – Electrodermal Activity).  
It combines **exploratory data analysis (EDA)** and **feature engineering** to prepare sensor signals for model training and evaluation.-->

---

## 🚀 Project Overview

<!-- Human Activity Recognition (HAR) is a fundamental problem in **wearable computing** and **IoT applications**,  
where the goal is to automatically classify activities such as *walking*, *sitting*, *running*, or *working* using motion and physiological data. --/>

In this project, we:
- Generate **synthetic sensor data** to simulate real human movement.
- Explore and visualize **Accelerometer** and **Gyroscope** signals.
- Compute **Euclidean norms** for motion magnitude estimation.
- Prepare data for future **ML/DL modeling** (classification).

---

## 📂 Dataset Structure

### Synthetic and Real Data

| Type | File | Description |
|------|------|--------------|
| 🧩 Training | `synthetic_training_data_20251014.csv` | Generated synthetic sensor data for training |
| 📘 Training | `training.csv` | Real or recorded sensor data (if available) |
| 🧩 Testing | `synthetic_test_data_20251014.csv` | Generated synthetic data for evaluation |
| 📗 Testing | `testing.csv` | Real test data for comparison |

Each dataset includes columns for:
- **Accelerometer**: `Acc_x`, `Acc_y`, `Acc_z`
- **Gyroscope**: `Gyro_x`, `Gyro_y`, `Gyro_z`
- (Optional) **EDA**, timestamps, and activity labels.

---

## 🧰 Features and Processing

### 1️⃣ Sensor Signal Normalization

#### 📈 Euclidean Distance of Accelerometer

$$
\mathrm{acc_{norm}}(i) = \| \mathbf{a}_i \|_2 = \sqrt{a_{i,x}^2 + a_{i,y}^2 + a_{i,z}^2}
$$

> The greater the acceleration, the faster the movement.

#### 🔄 Euclidean Distance of Gyroscope

$$
\mathrm{gyro_{norm}}(i) = \sqrt{\mathrm{Gyro_x}(i)^2 + \mathrm{Gyro_y}(i)^2 + \mathrm{Gyro_z}(i)^2}
$$

> The higher the gyroscope norm, the faster the rotation.

---

### 2️⃣ Definitions

| Term | Description |
|------|--------------|
| **Accelerometer** | Measures linear acceleration of the device in 3D space (m/s²). |
| **Gyroscope** | Measures angular velocity or rotational speed around each axis (rad/s). |
| **EDA (Electrodermal Activity)** | Measures skin conductance, often related to stress or workload. |

---

## 🧪 Environment Setup

### 1️⃣ Create Environment
```bash
conda create -n your_env python=3.11
conda activate your_env
conda install -n your_env ipykernel --update-deps --force-reinstall
### 2️⃣ Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

📚 References
- AIO Coffee Team
- [Human-Activity-Recognition – GitHub Repository](https://github.com/ma-shamshiri/Human-Activity-Recognition)
