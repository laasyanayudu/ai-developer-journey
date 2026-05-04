import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ============================================
# CONCEPT 2: LINEAR REGRESSION
# Predicting model speed based on accuracy
# ============================================

# Our dataset
data = {
    "accuracy": [75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97],
    "speed":    [20, 25, 30, 35, 45, 50, 60, 75, 85, 95, 110]
}

df = pd.DataFrame(data)

# Step 1 — Separate input (X) and output (y)
X = df[["accuracy"]]  # input  — what we know
y = df["speed"]       # output — what we want to predict

# Step 2 — Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Step 3 — Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)  # THIS is where learning happens

print(f"\nModel learned:")
print(f"  For every 1% accuracy increase → speed increases by {model.coef_[0]:.2f}")

# Step 4 — Make predictions
y_pred = model.predict(X_test)

print(f"\n--- Predictions vs Reality ---")
for actual, predicted in zip(y_test, y_pred):
    print(f"  Actual: {actual:>6} | Predicted: {predicted:.1f}")

# Step 5 — Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nModel error (RMSE): {rmse:.2f}")
print(f"(Lower is better — means predictions are off by ~{rmse:.0f} units on average)")

# Step 6 — Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="steelblue", label="Real data", zorder=5)
plt.plot(X, model.predict(X), color="red", label="Model prediction line")
plt.xlabel("Accuracy %")
plt.ylabel("Speed")
plt.title("Linear Regression — Accuracy vs Speed")
plt.legend()
plt.tight_layout()
plt.savefig("regression.png")
plt.show()
print("\nChart saved as regression.png!")
# ============================================
# CONCEPT 3: CLASSIFICATION
# Predicting if a model is "good" or "not good"
# based on accuracy and speed
# ============================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset — each model has accuracy, speed, and a label
data2 = {
    "accuracy": [75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97, 72, 99, 88, 76],
    "speed":    [20, 25, 30, 35, 45, 50, 60, 75, 85, 95,110, 15,120, 55, 22],
    "label":    [0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0]
    # label: 1 = good model, 0 = not good model
}

df2 = pd.DataFrame(data2)

# Step 1 — Separate input (X) and output (y)
X2 = df2[["accuracy", "speed"]]  # TWO inputs this time
y2 = df2["label"]

# Step 2 — Split data
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# Step 3 — Create and train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X2_train, y2_train)       # model learns the boundary

# Step 4 — Predict
y2_pred = clf.predict(X2_test)

# Step 5 — Evaluate
print("\n============================================")
print("CONCEPT 3: CLASSIFICATION RESULTS")
print("============================================")

print(f"\nAccuracy score: {accuracy_score(y2_test, y2_pred) * 100:.1f}%")

print("\n--- Predictions vs Reality ---")
for actual, predicted in zip(y2_test, y2_pred):
    label_actual = "Good model" if actual == 1 else "Not good"
    label_pred   = "Good model" if predicted == 1 else "Not good"
    match = "✓" if actual == predicted else "✗"
    print(f"  Actual: {label_actual:<12} | Predicted: {label_pred:<12} {match}")

print("\n--- Full Report ---")
print(classification_report(y2_test, y2_pred,
      target_names=["Not good", "Good model"]))

# Step 6 — Test with a brand new model nobody has seen before
print("--- Predict a brand new model ---")
new_model = [[92, 80]]   # accuracy=92, speed=80
prediction = clf.predict(new_model)
result = "Good model ✓" if prediction[0] == 1 else "Not good ✗"
print(f"  New model (accuracy=92, speed=80) → {result}")
# ============================================
# CONCEPT 4: MODEL EVALUATION
# Comparing multiple models to find the best one
# ============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("\n============================================")
print("CONCEPT 4: MODEL EVALUATION")
print("============================================")

# We'll use the same data as Concept 3
# X2_train, X2_test, y2_train, y2_test already exist

# --- Step 1: Try 3 different models ---
models = {
    "Decision Tree":   DecisionTreeClassifier(random_state=42),
    "Random Forest":   RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression()
}

print("\n--- Comparing 3 models ---")
best_score = 0
best_name = ""

for name, model in models.items():
    # Train
    model.fit(X2_train, y2_train)
    # Predict
    preds = model.predict(X2_test)
    # Score
    score = accuracy_score(y2_test, preds) * 100
    print(f"  {name:<25} → {score:.1f}%")

    # Track best
    if score > best_score:
        best_score = score
        best_name = name

print(f"\n🏆 Best model: {best_name} ({best_score:.1f}%)")

# --- Step 2: Confusion Matrix ---
# Shows exactly where your model gets confused
print("\n--- Confusion Matrix (Decision Tree) ---")
cm = confusion_matrix(y2_test, y2_pred)
print(cm)
print("""
What this means:
  [True Negatives,  False Positives]
  [False Negatives, True Positives ]
""")

# --- Step 3: Visualize the confusion matrix ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not good", "Good model"],
            yticklabels=["Not good", "Good model"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved!")

# --- Step 4: Cross validation ---
# Tests your model on MULTIPLE splits, not just one
from sklearn.model_selection import cross_val_score

print("\n--- Cross Validation (more reliable score) ---")
clf2 = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(clf2, X2, y2, cv=3)
print(f"  Scores across 3 splits: {[round(s*100,1) for s in scores]}")
print(f"  Average score: {scores.mean()*100:.1f}%")
print(f"  (More reliable than a single train/test split)")