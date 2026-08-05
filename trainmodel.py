# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📊 TRAINING SALARY PREDICTION MODEL")
print("=" * 60)

# ==================== LOAD DATA ====================
df = pd.read_csv('Salary Data.csv')
df = df.dropna(how='all')
print(f"\n✅ Loaded {len(df)} records")

# ==================== ENCODE CATEGORICAL VARIABLES ====================
print("\n🔧 Encoding categorical variables...")

# Use LabelEncoder instead of get_dummies to keep features small
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['Education_Encoded'] = le_education.fit_transform(df['Education Level'])
df['Job_Encoded'] = le_job.fit_transform(df['Job Title'])

# Select features (ONLY 5 features)
X = df[['Age', 'Gender_Encoded', 'Education_Encoded', 'Years of Experience', 'Job_Encoded']]
y = df['Salary']

print(f"✅ Features: {X.shape[1]} columns")
print(f"📋 Features: {X.columns.tolist()}")

# ==================== TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Training: {len(X_train)} samples")
print(f"✅ Testing: {len(X_test)} samples")

# ==================== SCALE FEATURES ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== TRAIN MODEL ====================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ==================== EVALUATE ====================
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE: ${mae:,.2f}")

# ==================== SAVE MODEL AND ENCODERS ====================
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_education, 'le_education.pkl')
joblib.dump(le_job, 'le_job.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("\n✅ Model saved successfully!")
print("📁 Files saved:")
print("  - model.pkl")
print("  - scaler.pkl")
print("  - le_gender.pkl")
print("  - le_education.pkl")
print("  - le_job.pkl")
print("  - feature_columns.pkl")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)