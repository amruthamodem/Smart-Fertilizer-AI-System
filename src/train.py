import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("🚀 Starting Training with 3 Models...\n")

# ======================================================
# 1️⃣ Load Dataset
# ======================================================
df = pd.read_csv("high_accuracy_fertilizer_dataset_15k.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("📂 Columns:", df.columns.tolist())

# Drop empty rows
df.dropna(how="all", inplace=True)

# ======================================================
# 2️⃣ Separate Features & Target
# ======================================================
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ======================================================
# 3️⃣ Identify Column Types
# ======================================================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("📊 Numerical Columns:", numerical_cols)
print("📊 Categorical Columns:", categorical_cols)

# ======================================================
# 4️⃣ Preprocessing
# Compatible with sklearn >= 1.2
# ======================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

# ======================================================
# 5️⃣ Train-Test Split
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

results = {}

# ======================================================
# 6️⃣ Decision Tree
# ======================================================
print("🌳 Training Decision Tree...")

dt_model = Pipeline([
    ("prep", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=42))
])

dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

print(f"Decision Tree Accuracy: {dt_acc:.4f}")
results["Decision Tree"] = (dt_acc, dt_model)

# ======================================================
# 7️⃣ Gradient Boosting
# ======================================================
print("\n🔥 Training Gradient Boosting...")

gb_model = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=42))
])

gb_model.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))

print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
results["Gradient Boosting"] = (gb_acc, gb_model)

# ======================================================
# 8️⃣ Support Vector Machine
# ======================================================
print("\n⚡ Training SVM...")

svm_model = Pipeline([
    ("prep", preprocessor),
    ("clf", SVC(kernel="rbf", C=100, probability=True))
])

svm_model.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

print(f"SVM Accuracy: {svm_acc:.4f}")
results["SVM"] = (svm_acc, svm_model)

# ======================================================
# 9️⃣ Select Best Model
# ======================================================
best_model_name = max(results, key=lambda k: results[k][0])
best_accuracy, best_model = results[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print(f"🎯 Best Accuracy: {best_accuracy:.4f}")

# ======================================================
# 🔟 Remove Old Model Files (Prevents Corruption)
# ======================================================
if os.path.exists("best_model.pkl"):
    os.remove("best_model.pkl")

if os.path.exists("target_encoder.pkl"):
    os.remove("target_encoder.pkl")

# ======================================================
# 1️⃣1️⃣ Save Model Using Joblib
# ======================================================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\n💾 Model and encoder saved successfully!")
print("✅ Training Completed.")