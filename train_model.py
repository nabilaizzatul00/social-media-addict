import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Students Social Media Addiction.csv")
df = df.drop(columns=["Student_ID", "Country"])
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.dropna()

# Ubah ke kategori
for col in ["Mental_Health_Score", "Conflicts_Over_Social_Media"]:
    df[col] = pd.cut(df[col], bins=[-1, 3, 6, 10], labels=["Low", "Medium", "High"])

# Pisahkan fitur dan target
X = df.drop(columns=["Addicted_Score"])
y = df["Addicted_Score"]

# Preprocessing
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipe, categorical),
    ("num", num_pipe, numerical)
])

# Final pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model_pipeline.pkl")
print("Model saved to model_pipeline.pkl")
