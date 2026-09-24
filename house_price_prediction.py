import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# BENGALURU HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

DATA_FILE = "Bengaluru_House_Data.csv"

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("BENGALURU HOUSE PRICE PREDICTION")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------
df = df.drop_duplicates().copy()

# Extract BHK number from the Size column
if "size" in df.columns:
    df["bhk"] = df["size"].astype(str).str.extract(r"(\d+)")[0]
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
else:
    raise ValueError("The dataset does not contain the expected 'size' column.")

# Convert total_sqft into a numerical value
def convert_sqft(value):
    try:
        value = str(value).strip()

        # Handle ranges such as 1200-1500
        if "-" in value:
            parts = value.split("-")
            numbers = []

            for part in parts:
                number = pd.to_numeric(part.strip(), errors="coerce")
                if pd.notna(number):
                    numbers.append(float(number))

            if numbers:
                return sum(numbers) / len(numbers)

        # Convert square metres to square feet
        if "Sq. Meter" in value:
            number = pd.to_numeric(
                value.replace("Sq. Meter", "").strip(),
                errors="coerce"
            )
            if pd.notna(number):
                return float(number) * 10.7639

        # Convert acres to square feet
        if "Acres" in value:
            number = pd.to_numeric(
                value.replace("Acres", "").strip(),
                errors="coerce"
            )
            if pd.notna(number):
                return float(number) * 43560

        return pd.to_numeric(value, errors="coerce")

    except Exception:
        return np.nan


df["total_sqft"] = df["total_sqft"].apply(convert_sqft)

# Convert numerical columns
df["bath"] = pd.to_numeric(df["bath"], errors="coerce")
df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Remove rows where essential values are missing
df = df.dropna(
    subset=["location", "total_sqft", "bath", "price", "bhk"]
).copy()

# Columns not required for the final model
columns_to_drop = ["area_type", "availability", "society", "size"]

for column in columns_to_drop:
    if column in df.columns:
        df = df.drop(column, axis=1)

# Remove clearly invalid/extreme records
df = df[
    (df["total_sqft"] > 200)
    & (df["total_sqft"] < 50000)
    & (df["price"] > 0)
    & (df["bhk"] > 0)
    & (df["bath"] > 0)
].copy()

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nBasic statistics:")
print(df.describe())

# Create folder for generated charts
output_dir = "project_outputs"
Path(output_dir).mkdir(exist_ok=True)

# House price distribution
plt.figure(figsize=(10, 6))
plt.hist(df["price"], bins=40, edgecolor="black")
plt.title("Distribution of House Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of Properties")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/house_price_distribution.png",
    dpi=150
)
plt.show()
plt.close()

# Total square feet vs price
plt.figure(figsize=(10, 6))
plt.scatter(
    df["total_sqft"],
    df["price"],
    alpha=0.5
)
plt.title("Total Square Feet vs House Price")
plt.xlabel("Total Square Feet")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/sqft_vs_price.png",
    dpi=150
)
plt.show()
plt.close()

# BHK vs price
plt.figure(figsize=(10, 6))
df.boxplot(
    column="price",
    by="bhk"
)
plt.suptitle("")
plt.title("BHK vs House Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (Lakhs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    f"{output_dir}/bhk_vs_price.png",
    dpi=150
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 4. FEATURES AND TARGET
# ------------------------------------------------------------
X = df[
    ["location", "total_sqft", "bath", "balcony", "bhk"]
]

y = df["price"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nTarget variable: price")

# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 6. PREPROCESSING PIPELINE
# ------------------------------------------------------------
numeric_features = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk"
]

categorical_features = ["location"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------
# 7. MACHINE LEARNING MODEL
# ------------------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

print("\nTraining Linear Regression model...")
model.fit(X_train, y_train)

print("Model training completed!")

# ------------------------------------------------------------
# 8. PREDICTIONS
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 9. MODEL EVALUATION
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
print(f"R² Score: {r2:.4f}")

# ------------------------------------------------------------
# 10. ACTUAL VS PREDICTED PRICE
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.savefig(
    f"{output_dir}/actual_vs_predicted.png",
    dpi=150
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 11. SAMPLE HOUSE PRICE PREDICTION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("SAMPLE HOUSE PRICE PREDICTION")
print("=" * 60)

sample_location = df["location"].iloc[0]

sample_house = pd.DataFrame({
    "location": [sample_location],
    "total_sqft": [1200],
    "bath": [2],
    "balcony": [1],
    "bhk": [2]
})

sample_prediction = model.predict(sample_house)[0]

print("\nSample property:")
print("Location:", sample_location)
print("Total Square Feet: 1200")
print("Bathrooms: 2")
print("Balcony: 1")
print("BHK: 2")

print(
    f"\nPredicted House Price: {sample_prediction:.2f} Lakhs"
)

# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Rows used after preprocessing:", df.shape[0])
print("Number of input features:", X.shape[1])
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} Lakhs")

print("\nGenerated charts are saved in the 'project_outputs' folder.")
print("Bengaluru House Price Prediction completed!")
