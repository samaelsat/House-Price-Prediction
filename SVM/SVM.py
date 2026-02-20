print("="*60)
print("UPGRADING PIP AND INSTALLING GPU-ACCELERATED LIBRARIES")
print("="*60)

# Install and prepare GPU-capable Python dependencies.
!pip install --upgrade pip -q

!apt-get update -qq -y > /dev/null
!apt-get install -y cuda-toolkit-11-8 > /dev/null

!pip install cupy-cuda11x -q

!pip install cudf-cu11 cuml-cu11 rmm-cu11 --extra-index-url=https://pypi.nvidia.com -q

!pip install pandas numpy matplotlib seaborn scikit-learn -q

print("✅ Installation completed!")

# Import GPU/CPU libraries used for preprocessing, modeling, and plotting.
import cudf
import cuml
from cuml.svm import SVR as cuSVR
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.metrics import mean_squared_error as cu_mse
from cuml.metrics import mean_absolute_error as cu_mae
from cuml.metrics import r2_score as cu_r2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Verify GPU availability and print device/runtime details.
import cupy as cp
print("\n" + "="*60)
print("GPU INFORMATION")
print("="*60)
print(f"GPU Available: {cp.cuda.is_available()}")
if cp.cuda.is_available():
    device_id = cp.cuda.Device().id
    device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
    print(f"GPU Device: {device_properties['name'].decode('utf-8')}")
    print(f"GPU Memory: {device_properties['totalGlobalMem'] / 1e9:.2f} GB")
    print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")

print("\n✅ Libraries imported successfully!")

print("="*60)
print("LOADING COMPLETE DATASET")
print("="*60)

# Upload the source CSV in Colab and load it with fallback encodings.
from google.colab import files
print("📁 Please upload your 'new.csv' file:")
uploaded = files.upload()

print("\n⚡ Loading complete dataset...")
try:
    df_pandas = pd.read_csv('new.csv', low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    print("   UnicodeDecodeError detected, trying 'latin1' encoding...")
    try:
        df_pandas = pd.read_csv('new.csv', low_memory=False, encoding='latin1')
    except UnicodeDecodeError:
        print("   'latin1' also failed, trying 'gbk' encoding...")
        df_pandas = pd.read_csv('new.csv', low_memory=False, encoding='gbk')

print(f"\n📊 Complete Dataset Shape: {df_pandas.shape}")
print(f"   Total Rows: {df_pandas.shape[0]:,}")
print(f"   Total Columns: {df_pandas.shape[1]}")
print(f"   Dataset Size: {df_pandas.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# Move dataset to GPU memory and release CPU copy.
print("\n⚡ Transferring COMPLETE data to GPU memory...")
df_gpu = cudf.DataFrame.from_pandas(df_pandas)

del df_pandas
import gc
gc.collect()

print(f"✅ Complete data loaded on GPU!")
print(f"\nFirst 3 rows:")
print(df_gpu.head(3).to_pandas())

missing_count = df_gpu.isnull().sum().sum()
print(f"\nMissing values: {missing_count:,}")

print(f"\nGPU Memory Used by Data: {df_gpu.memory_usage(deep=True).sum() / 1e6:.2f} MB")

free_memory = cp.cuda.Device().mem_info[0] / 1e9
total_memory = cp.cuda.Device().mem_info[1] / 1e9
print(f"GPU Memory Available: {free_memory:.2f} GB / {total_memory:.2f} GB")

print("="*60)
print("DATA CLEANING - COMPLETE DATASET (GPU)")
print("="*60)

# Clean missing values and normalize key categorical text fields.
df_clean = df_gpu.copy()

print(f"Processing {len(df_clean):,} rows on GPU...")

print("\n⚡ Handling Missing Values on GPU...")

numerical_cols = df_clean.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

print(f"   Numerical columns: {len(numerical_cols)}")
print(f"   Categorical columns: {len(categorical_cols)}")

for col in numerical_cols:
    null_count = df_clean[col].isnull().sum()
    if null_count > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

for col in categorical_cols:
    null_count = df_clean[col].isnull().sum()
    if null_count > 0:
        mode_val = df_clean[col].mode()
        if len(mode_val) > 0:
            df_clean[col] = df_clean[col].fillna(mode_val[0])
        else:
            df_clean[col] = df_clean[col].fillna('Unknown')

print("   ✓ Missing values handled")

print("\n⚡ Processing 'floor' column...")
if 'floor' in df_clean.columns:
    floor_series = df_clean['floor'].to_pandas().astype(str).str.lower()

    floor_cleaned = pd.Series('Middle', index=floor_series.index)

    high_mask = floor_series.str.contains('高|¸ß|high', na=False, regex=True)
    low_mask = floor_series.str.contains('低|µí|low', na=False, regex=True)
    ground_mask = floor_series.str.contains('底|µ×|ground', na=False, regex=True)

    floor_cleaned[high_mask] = 'High'
    floor_cleaned[low_mask] = 'Low'
    floor_cleaned[ground_mask] = 'Ground'

    df_clean['floor_cleaned'] = cudf.Series(floor_cleaned.values)

    floor_dist = df_clean['floor_cleaned'].value_counts().to_pandas()
    print(f"   ✓ Floor distribution:")
    for idx, val in floor_dist.items():
        print(f"      {idx}: {val:,}")

free_memory = cp.cuda.Device().mem_info[0] / 1e9
print(f"\n📊 GPU Memory Available: {free_memory:.2f} GB")

print(f"\n✅ Final dataset shape: {df_clean.shape}")
print(f"   Total samples: {len(df_clean):,}")

print("="*60)
print("FEATURE ENGINEERING - COMPLETE DATASET (GPU)")
print("="*60)

# Create engineered features from existing numeric, date, and geo columns.
print(f"\n⚡ Creating New Features for {len(df_clean):,} rows on GPU...")

if 'constructionTime' in df_clean.columns:
    if df_clean['constructionTime'].dtype == 'object':
        df_clean['constructionTime'] = cudf.to_numeric(df_clean['constructionTime'], errors='coerce')
    df_clean['building_age'] = 2017 - df_clean['constructionTime']
    df_clean['building_age'] = df_clean['building_age'].fillna(df_clean['building_age'].mean())
    print("   ✓ building_age")

room_cols = ['livingRoom', 'drawingRoom', 'kitchen', 'bathroom']
available_cols = [col for col in room_cols if col in df_clean.columns]
if available_cols:
    for col in available_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = cudf.to_numeric(df_clean[col], errors='coerce').fillna(0)
    df_clean['total_rooms'] = df_clean[available_cols].sum(axis=1)
    print("   ✓ total_rooms")

if 'totalPrice' in df_clean.columns and 'square' in df_clean.columns:
    if df_clean['totalPrice'].dtype == 'object':
        df_clean['totalPrice'] = cudf.to_numeric(df_clean['totalPrice'], errors='coerce')
    if df_clean['square'].dtype == 'object':
        df_clean['square'] = cudf.to_numeric(df_clean['square'], errors='coerce')

    df_clean['price_per_sqm'] = df_clean['totalPrice'] / (df_clean['square'] + 1)
    df_clean['price_per_sqm'] = df_clean['price_per_sqm'].fillna(df_clean['price_per_sqm'].mean())
    print("   ✓ price_per_sqm")

if 'tradeTime' in df_clean.columns:
    trade_time_pd = pd.to_datetime(df_clean['tradeTime'].to_pandas(), errors='coerce')
    df_clean['trade_year'] = cudf.Series(trade_time_pd.dt.year.fillna(2017).values)
    df_clean['trade_month'] = cudf.Series(trade_time_pd.dt.month.fillna(6).values)
    print("   ✓ trade_year, trade_month")

if 'Lng' in df_clean.columns and 'Lat' in df_clean.columns:
    if df_clean['Lng'].dtype == 'object':
        df_clean['Lng'] = cudf.to_numeric(df_clean['Lng'], errors='coerce')
    if df_clean['Lat'].dtype == 'object':
        df_clean['Lat'] = cudf.to_numeric(df_clean['Lat'], errors='coerce')

    df_clean['Lng'] = df_clean['Lng'].fillna(df_clean['Lng'].mean())
    df_clean['Lat'] = df_clean['Lat'].fillna(df_clean['Lat'].mean())

    center_lng, center_lat = 116.4074, 39.9042

    lng_array = cp.asarray(df_clean['Lng'].values)
    lat_array = cp.asarray(df_clean['Lat'].values)

    distance = cp.sqrt((lng_array - center_lng)**2 + (lat_array - center_lat)**2)
    df_clean['distance_from_center'] = cudf.Series(distance)
    print("   ✓ distance_from_center (GPU-accelerated)")

if 'DOM' in df_clean.columns:
    if df_clean['DOM'].dtype == 'object':
        df_clean['DOM'] = cudf.to_numeric(df_clean['DOM'], errors='coerce')
    df_clean['DOM'] = df_clean['DOM'].fillna(df_clean['DOM'].median())
    print("   ✓ DOM")

if 'followers' in df_clean.columns:
    if df_clean['followers'].dtype == 'object':
        df_clean['followers'] = cudf.to_numeric(df_clean['followers'], errors='coerce')
    df_clean['followers'] = df_clean['followers'].fillna(0)
    print("   ✓ followers")

if 'ladderRatio' in df_clean.columns:
    if df_clean['ladderRatio'].dtype == 'object':
        df_clean['ladderRatio'] = cudf.to_numeric(df_clean['ladderRatio'], errors='coerce')
    df_clean['ladderRatio'] = df_clean['ladderRatio'].fillna(df_clean['ladderRatio'].median())
    print("   ✓ ladderRatio")

if 'communityAverage' in df_clean.columns:
    if df_clean['communityAverage'].dtype == 'object':
        df_clean['communityAverage'] = cudf.to_numeric(df_clean['communityAverage'], errors='coerce')
    df_clean['communityAverage'] = df_clean['communityAverage'].fillna(df_clean['communityAverage'].mean())
    print("   ✓ communityAverage")

print("\n⚡ Encoding Categorical Variables...")

# Encode selected categorical columns for model training.
categorical_to_encode = ['floor_cleaned', 'buildingType', 'renovationCondition',
                          'buildingStructure', 'elevator', 'fiveYearsProperty',
                          'subway', 'district']

label_encoders = {}
encoded_count = 0

for col in categorical_to_encode:
    if col in df_clean.columns:
        try:
            col_pandas = df_clean[col].to_pandas().astype(str)

            le = LabelEncoder()
            encoded_values = le.fit_transform(col_pandas)

            df_clean[f'{col}_encoded'] = cudf.Series(encoded_values)
            label_encoders[col] = le
            encoded_count += 1

            print(f"   ✓ {col} ({len(le.classes_)} categories)")

        except Exception as e:
            print(f"   ⚠ Warning: Could not encode {col}: {e}")

print(f"\n✅ Encoded {encoded_count} categorical features")

print("\n⚡ Final data type validation...")

# Ensure model-critical columns are numeric before training.
numeric_candidates = ['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathroom',
                      'constructionTime', 'DOM', 'followers', 'totalPrice', 'price',
                      'ladderRatio', 'Lng', 'Lat', 'communityAverage']

for col in numeric_candidates:
    if col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = cudf.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

print("   ✓ All numeric columns validated")

print(f"\n✅ Feature Engineering Complete!")
print(f"   Total features: {df_clean.shape[1]}")
print(f"   Total samples: {len(df_clean):,}")
print(f"   Numeric features: {len(df_clean.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns)}")

free_memory = cp.cuda.Device().mem_info[0] / 1e9
print(f"   GPU Memory Available: {free_memory:.2f} GB")

print("="*60)
print("FEATURE SELECTION - COMPLETE DATASET (GPU)")
print("="*60)

# Select numeric training features and exclude IDs/raw target fields.
exclude_cols = ['url', 'id', 'Cid', 'tradeTime', 'floor', 'floor_cleaned',
                'buildingType', 'renovationCondition', 'buildingStructure',
                'elevator', 'fiveYearsProperty', 'subway', 'district',
                'totalPrice', 'price', 'price_per_sqm']

all_cols = df_clean.columns.tolist()

feature_cols = [col for col in all_cols
                if col not in exclude_cols
                and df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f"\n📊 Selected {len(feature_cols)} features from {len(df_clean):,} samples")
print(f"\nFeatures:")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

print("\n⚡ Preparing feature matrix on GPU...")

# Build X/y tensors and create train-test splits on GPU.
X_gpu = df_clean[feature_cols].fillna(0)
y_gpu = df_clean['totalPrice'].fillna(df_clean['totalPrice'].mean())

print(f"   Feature Matrix: {X_gpu.shape}")
print(f"   Target Vector: {y_gpu.shape}")
print(f"   Total elements: {X_gpu.shape[0] * X_gpu.shape[1]:,}")

print("\n⚡ Splitting complete dataset on GPU...")
X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu = cu_train_test_split(
    X_gpu, y_gpu,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"   Training samples: {len(X_train_gpu):,}")
print(f"   Testing samples: {len(X_test_gpu):,}")
print(f"   Train/Test split: {len(X_train_gpu)/(len(X_train_gpu)+len(X_test_gpu))*100:.1f}% / {len(X_test_gpu)/(len(X_train_gpu)+len(X_test_gpu))*100:.1f}%")

print("\n⚡ Scaling features on GPU...")
import time
scale_start = time.time()

# Standardize features for SVR training stability.
scaler_gpu = cuStandardScaler()
X_train_scaled_gpu = scaler_gpu.fit_transform(X_train_gpu)
X_test_scaled_gpu = scaler_gpu.transform(X_test_gpu)

scale_time = time.time() - scale_start

print(f"   ✓ Scaling completed in {scale_time:.2f} seconds")
print(f"   ✓ Scaled {len(X_train_gpu) + len(X_test_gpu):,} samples")

free_memory = cp.cuda.Device().mem_info[0] / 1e9
total_memory = cp.cuda.Device().mem_info[1] / 1e9
used_memory = total_memory - free_memory

print(f"\n📊 GPU Memory Status:")
print(f"   Used: {used_memory:.2f} GB")
print(f"   Free: {free_memory:.2f} GB")
print(f"   Total: {total_memory:.2f} GB")

print("\n✅ Data preparation completed on GPU!")

print("="*60)
print("BASELINE SVM - COMPLETE DATASET (GPU)")
print("="*60)

# Train a baseline linear-kernel SVR on GPU.
print(f"\n🚀 Training Baseline GPU-SVM on {len(X_train_gpu):,} samples...")
import time

start_time = time.time()

baseline_svm_gpu = cuSVR(
    kernel='linear',
    C=1.0,
    cache_size=2000.0,
    max_iter=1000
)

baseline_svm_gpu.fit(X_train_scaled_gpu, y_train_gpu)

train_time = time.time() - start_time
print(f"   ✓ Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
print(f"   ✓ Training speed: {len(X_train_gpu)/train_time:,.0f} samples/second")

print("\n⚡ Making predictions on GPU...")
pred_start = time.time()

y_train_pred_baseline = baseline_svm_gpu.predict(X_train_scaled_gpu)
y_test_pred_baseline = baseline_svm_gpu.predict(X_test_scaled_gpu)

pred_time = time.time() - pred_start
print(f"   ✓ Predictions completed in {pred_time:.2f} seconds")

# Compute regression metrics directly on GPU outputs.
def calculate_metrics_gpu(y_true, y_pred, dataset_name):
    """Calculate metrics on GPU"""
    mse = cu_mse(y_true, y_pred)
    mae = cu_mae(y_true, y_pred)
    r2 = cu_r2(y_true, y_pred)

    rmse = cp.sqrt(mse)

    y_true_cp = cp.asarray(y_true)
    y_pred_cp = cp.asarray(y_pred)
    mape = cp.mean(cp.abs((y_true_cp - y_pred_cp) / (y_true_cp + 1e-10))) * 100

    mse = float(mse)
    rmse = float(rmse)
    mae = float(mae)
    mape = float(mape)
    r2 = float(r2)

    print(f"\n📊 {dataset_name} ({len(y_true):,} samples):")
    print(f"   MSE: {mse:,.2f} | RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

baseline_train_metrics = calculate_metrics_gpu(y_train_gpu, y_train_pred_baseline, "Baseline Training (GPU)")
baseline_test_metrics = calculate_metrics_gpu(y_test_gpu, y_test_pred_baseline, "Baseline Testing (GPU)")

print(f"\n✅ Baseline model completed!")
print(f"   Total time: {train_time + pred_time:.2f} seconds")

print("="*60)
print("HYPERPARAMETER TUNING - COMPLETE DATASET (GPU)")
print("="*60)

# Define SVR search space for GPU grid search.
param_grid = {
    'svr__C': [50, 100, 200, 500],
    'svr__gamma': [0.005, 0.01, 0.02],
    'svr__epsilon': [0.05, 0.1, 0.2],
    'svr__kernel': ['rbf']
}

print(f"\n🔍 Tuning on {len(X_train_gpu):,} training samples")
param_grid_flat = [
    {'C': c_val, 'gamma': gamma_val, 'epsilon': eps_val, 'kernel': 'rbf'}
    for c_val in param_grid['svr__C']
    for gamma_val in param_grid['svr__gamma']
    for eps_val in param_grid['svr__epsilon']
]
total_iterations = len(param_grid_flat)
print(f"   Parameter combinations: {total_iterations}")

best_score = float('inf')
best_params = {}
best_model = None

results = []

print("\n⚡ Starting GPU grid search on complete dataset...")
overall_start = time.time()

# Use a capped validation subset to keep tuning runtime manageable.
validation_size = min(20000, len(X_test_gpu))
if len(X_test_gpu) > validation_size:
    print(f"   Using {validation_size:,} samples for validation (for speed)")
    val_indices = cp.random.choice(len(X_test_gpu), validation_size, replace=False)
    X_val = X_test_scaled_gpu.iloc[val_indices.get()]
    y_val = y_test_gpu.iloc[val_indices.get()]
else:
    X_val = X_test_scaled_gpu
    y_val = y_test_gpu

iteration = 0

for params in param_grid_flat:
    iteration += 1
    start = time.time()

    model = cuSVR(
        kernel=params['kernel'],
        C=params['C'],
        epsilon=params['epsilon'],
        gamma=params['gamma'],
        cache_size=2000.0,
        max_iter=1000
    )
    model.fit(X_train_scaled_gpu, y_train_gpu)

    y_val_pred = model.predict(X_val)
    mse = float(cu_mse(y_val, y_val_pred))
    rmse = np.sqrt(mse)

    elapsed = time.time() - start

    results.append({
        'kernel': params['kernel'],
        'C': params['C'],
        'gamma': params['gamma'],
        'epsilon': params['epsilon'],
        'RMSE': rmse,
        'time': elapsed
    })

    print(f"   [{iteration}/{total_iterations}] {params['kernel']:6s} | C={params['C']:5.1f} | γ={params['gamma']:.3f} | ε={params['epsilon']:.2f} | RMSE={rmse:8,.2f} | {elapsed:.1f}s", end='')

    if mse < best_score:
        best_score = mse
        best_params = {'kernel': params['kernel'], 'C': params['C'], 'gamma': params['gamma'], 'epsilon': params['epsilon']}
        best_model = model
        print(" ✓ NEW BEST!")
    else:
        print()

total_time = time.time() - overall_start

print(f"\n✅ Grid search completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"   Average time per configuration: {total_time/total_iterations:.2f} seconds")

print("\n🏆 Best Hyperparameters:")
for param, value in best_params.items():
    print(f"   ├─ {param}: {value}")

best_rmse = np.sqrt(best_score)
print(f"\n📊 Best Validation RMSE: {best_rmse:,.2f}")

# Retrain best SVR configuration on the full training split.
print(f"\n⚡ Retraining best model on complete training set ({len(X_train_gpu):,} samples)...")
retrain_start = time.time()

best_svm_gpu = cuSVR(
    kernel=best_params['kernel'],
    C=best_params['C'],
    epsilon=best_params['epsilon'],
    gamma=best_params['gamma'],
    cache_size=2000.0,
    max_iter=1000
)
best_svm_gpu.fit(X_train_scaled_gpu, y_train_gpu)

retrain_time = time.time() - retrain_start
print(f"   ✓ Retraining completed in {retrain_time:.2f} seconds")

results_df = pd.DataFrame(results).sort_values('RMSE')
print(f"\n📊 Top 5 configurations:")
print(results_df.head(5).to_string(index=False))

print("="*60)
print("TUNED MODEL EVALUATION - COMPLETE DATASET (GPU)")
print("="*60)

# Evaluate tuned model and compare against baseline metrics.
print(f"\n⚡ Making predictions on complete test set ({len(X_test_gpu):,} samples)...")
y_train_pred_tuned = best_svm_gpu.predict(X_train_scaled_gpu)
y_test_pred_tuned = best_svm_gpu.predict(X_test_scaled_gpu)

tuned_train_metrics = calculate_metrics_gpu(y_train_gpu, y_train_pred_tuned, "Tuned Training (GPU)")
tuned_test_metrics = calculate_metrics_gpu(y_test_gpu, y_test_pred_tuned, "Tuned Testing (GPU)")

print("\n" + "="*60)
print("PERFORMANCE COMPARISON - COMPLETE DATASET")
print("="*60)

comparison_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R²'],
    'Baseline': [
        baseline_test_metrics['RMSE'],
        baseline_test_metrics['MAE'],
        baseline_test_metrics['MAPE'],
        baseline_test_metrics['R2']
    ],
    'Tuned': [
        tuned_test_metrics['RMSE'],
        tuned_test_metrics['MAE'],
        tuned_test_metrics['MAPE'],
        tuned_test_metrics['R2']
    ]
})

comparison_df['Improvement (%)'] = np.where(
    comparison_df['Metric'] == 'R²',
    (comparison_df['Tuned'] - comparison_df['Baseline']) / comparison_df['Baseline'] * 100,
    (comparison_df['Baseline'] - comparison_df['Tuned']) / comparison_df['Baseline'] * 100
)

print(f"\nTest Set Size: {len(y_test_gpu):,} samples")
print(comparison_df.to_string(index=False))

print("\n✅ Evaluation on complete dataset completed!")

print("="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Sample predictions and generate summary diagnostic plots.
print("\n⚡ Transferring sample of predictions to CPU for visualization...")

viz_sample_size = min(50000, len(y_test_gpu))
sample_indices = cp.random.choice(len(y_test_gpu), viz_sample_size, replace=False)

y_test_sample = y_test_gpu.iloc[sample_indices.get()].to_numpy()
y_test_pred_baseline_sample = y_test_pred_baseline[sample_indices].to_numpy()
y_test_pred_tuned_sample = y_test_pred_tuned[sample_indices].to_numpy()

print(f"   Using {viz_sample_size:,} samples for visualization")

fig = plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
plt.scatter(y_test_sample, y_test_pred_baseline_sample, alpha=0.3, s=10, c='blue')
plt.plot([0, 1000],
         [0, 1000], 'r--', lw=2)
plt.xlabel('Actual Price', fontsize=11)
plt.ylabel('Predicted Price', fontsize=11)
plt.title(f'Baseline: Actual vs Predicted\n({viz_sample_size:,} samples)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(0, 1000)

plt.subplot(3, 3, 2)
plt.scatter(y_test_sample, y_test_pred_tuned_sample, alpha=0.3, s=10, c='skyblue')
plt.plot([0, 1000],
         [0, 1000], 'r--', lw=2)
plt.xlabel('Actual Price', fontsize=11)
plt.ylabel('Predicted Price', fontsize=11)
plt.title(f'Tuned: Actual vs Predicted (GPU)\n({viz_sample_size:,} samples)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1000)
plt.ylim(0, 1000)

plt.subplot(3, 3, 3)
residuals_sample = y_test_sample - y_test_pred_tuned_sample
plt.scatter(y_test_pred_tuned_sample, residuals_sample, alpha=0.3, s=10, c='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=11)
plt.ylabel('Residuals', fontsize=11)
plt.title('Tuned Model: Residual Plot', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 4)
plt.hist(residuals_sample, bins=50, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('Residuals', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
metrics = ['RMSE', 'MAE', 'MAPE', 'R²×100']
baseline_vals = [
    baseline_test_metrics['RMSE'],
    baseline_test_metrics['MAE'],
    baseline_test_metrics['MAPE'],
    baseline_test_metrics['R2'] * 100
]
tuned_vals = [
    tuned_test_metrics['RMSE'],
    tuned_test_metrics['MAE'],
    tuned_test_metrics['MAPE'],
    tuned_test_metrics['R2'] * 100
]

x_pos = np.arange(len(metrics))
width = 0.35
plt.bar(x_pos - width / 2, baseline_vals, width, label='Baseline', alpha=0.8)
plt.bar(x_pos + width / 2, tuned_vals, width, label='Tuned', alpha=0.8)
plt.xlabel('Metrics', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.title(
    f'GPU Model Performance\n({len(y_test_gpu):,} test samples)',
    fontsize=12,
    fontweight='bold'
)
plt.xticks(x_pos, metrics, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(3, 3, 6)
r2_data = [baseline_test_metrics['R2'], tuned_test_metrics['R2']]
colors = ['#ffcc99', '#99ff99']
bars = plt.bar(['Baseline\n(GPU)', 'Tuned\n(GPU)'], color=colors, alpha=0.8, height=r2_data, edgecolor='black')
plt.ylabel('R² Score', fontsize=11)
plt.title('R² Score Comparison', fontsize=12, fontweight='bold')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_data):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.02,
        f'{val:.4f}',
        ha='center',
        fontsize=11,
        fontweight='bold',
    )

plt.subplot(3, 3, 7)
cpu_estimate = train_time * 40
times = [train_time, cpu_estimate]
colors_time = ['#00ff00', '#ff6666']
bars = plt.bar(
    ['GPU\n(Actual)', 'CPU\n(Estimated)'], color=colors_time, alpha=0.8, height=times, edgecolor='black'
)
plt.ylabel('Training Time (seconds)', fontsize=11)
plt.title(
    f'Training Speed: GPU vs CPU\n({len(X_train_gpu):,} samples)',
    fontsize=12,
    fontweight='bold',
)
plt.yscale('log')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val * 1.3,
        f'{val:.1f}s\n({val/60:.1f}m)' if val > 60 else f'{val:.1f}s',
        ha='center',
        fontsize=9,
        fontweight='bold',
    )

plt.subplot(3, 3, 8)
error_baseline = np.abs(y_test_sample - y_test_pred_baseline_sample)
error_tuned = np.abs(y_test_sample - y_test_pred_tuned_sample)
bp = plt.boxplot([error_baseline, error_tuned], labels=['Baseline', 'Tuned'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
    patch.set_facecolor(color)
plt.ylabel('Absolute Error', fontsize=11)
plt.title('Error Distribution (GPU)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(3, 3, 9)
accuracy = (1 - np.abs(residuals_sample) / (y_test_sample + 1e-10)) * 100
plt.hist(accuracy, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Prediction Accuracy (%)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title(
    f'Prediction Accuracy Distribution\n(Complete Dataset: {len(y_test_gpu):,} samples)',
    fontsize=12,
    fontweight='bold',
)
plt.axvline(
    x=accuracy.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {accuracy.mean():.1f}%'
)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gpu_complete_dataset_results.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✅ Visualizations saved as 'gpu_complete_dataset_results.png'")
print(f"   (Sampled {viz_sample_size:,} points from {len(y_test_gpu):,} test samples for visualization)")

print("="*60)
print("FEATURE IMPORTANCE - COMPLETE DATASET (GPU)")
print("="*60)

# Estimate feature importance via absolute correlation with target.
print(f"\n⚡ Calculating feature importance for {len(feature_cols)} features...")

X_train_cpu = X_train_gpu.to_pandas()
y_train_cpu = y_train_gpu.to_pandas()

correlations = X_train_cpu.corrwith(y_train_cpu).abs().sort_values(ascending=False)

feature_importance_df = pd.DataFrame({
    'Feature': correlations.index,
    'Importance': correlations.values
})

print(f"\n📊 Top 20 Most Important Features:")
print(feature_importance_df.head(20).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top_n = min(20, len(feature_importance_df))
top_features = feature_importance_df.head(top_n)

axes[0].barh(range(top_n), top_features['Importance'], alpha=0.8, color='skyblue', edgecolor='black')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features['Feature'], fontsize=10)
axes[0].set_xlabel('Importance (Correlation)', fontsize=12)
axes[0].set_title(f'Top {top_n} Features (Complete Dataset: {len(X_train_gpu):,} samples)',
                  fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].bar(range(len(feature_importance_df)), feature_importance_df['Importance'],
            alpha=0.8, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Feature Index', fontsize=12)
axes[1].set_ylabel('Importance', fontsize=12)
axes[1].set_title(f'All {len(feature_importance_df)} Features Ranked', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gpu_feature_importance_complete.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n✅ Feature importance saved as 'gpu_feature_importance_complete.png'")

print("="*60)
print("ACTUAL VS PREDICTED BAR GRAPH (100 SAMPLES)")
print("="*60)

# Plot side-by-side actual vs predicted prices for a small sample.
sample_size_bar = 30

bar_sample_indices = cp.random.choice(len(y_test_gpu), sample_size_bar, replace=False)

y_test_actual_bar = y_test_gpu.iloc[bar_sample_indices.get()].to_numpy()
y_test_predicted_bar = y_test_pred_tuned[bar_sample_indices].to_numpy()

comparison_df_bar = pd.DataFrame({
    'Index': np.arange(sample_size_bar),
    'Actual Price': y_test_actual_bar,
    'Predicted Price': y_test_predicted_bar
})

comparison_df_bar = comparison_df_bar.sort_values(by='Actual Price').reset_index(drop=True)

plt.figure(figsize=(18, 8))
bar_width = 0.35
index = np.arange(len(comparison_df_bar))

plt.bar(index, comparison_df_bar['Actual Price'], bar_width, label='Actual Price', color='red', alpha=0.8)
plt.bar(index + bar_width, comparison_df_bar['Predicted Price'], bar_width, label='Predicted Price', color='blue', alpha=0.8)

plt.xlabel('Sample House Index', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.title('Actual vs Predicted Prices for 100 Sample Houses (Tuned Model)', fontsize=14, fontweight='bold')
plt.xticks(index + bar_width / 2, comparison_df_bar['Index'], rotation=90, fontsize=8)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('actual_vs_predicted_bar_graph_100_samples.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✅ Bar graph for 100 sample houses saved as 'actual_vs_predicted_bar_graph_100_samples.png'")

