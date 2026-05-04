import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# PHASE 2 PROJECT: MOVIE RATING PREDICTOR
# By: Chetana Laasya Nayudu
# ============================================


print("=============================")
print("   MOVIE RATING PREDICTOR   ")
print("=============================")

df = pd.read_csv("tmdb_5000_movies.csv")

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df[["title", "vote_average", "budget", "revenue", "runtime"]].head(3))
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Cleaning data ---")

df = df.drop(columns=["homepage", "tagline", "keywords",
                       "overview", "genres", "production_companies",
                       "production_countries", "spoken_languages",
                       "original_title", "id"])

df = df.dropna()


df = df[["budget", "revenue", "runtime", "popularity",
         "vote_count", "vote_average", "original_language"]]


df = df[df["original_language"] == "en"]
df = df.drop(columns=["original_language"])

df = df[df["budget"] > 0]
df = df[df["revenue"] > 0]

print(f"Rows after cleaning: {len(df)}")
print(df.describe().round(2))


print("\n--- Creating target label ---")


df["high_rated"] = (df["vote_average"] >= 7.0).astype(int)

print(f"High rated movies (≥7.0): {df['high_rated'].sum()}")
print(f"Low rated movies  (<7.0): {(df['high_rated']==0).sum()}")


plt.figure(figsize=(6, 4))
df["high_rated"].value_counts().plot(kind="bar",
    color=["steelblue", "green"])
plt.xticks([0, 1], ["Low rated", "High rated"], rotation=0)
plt.title("Movie Rating Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()
print("Chart saved!")

print("\n--- Preparing features ---")

X = df[["budget", "revenue", "runtime", "popularity", "vote_count"]]
y = df["high_rated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


print("\n--- Training and comparing models ---")

models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    # Score
    score = accuracy_score(y_test, preds) * 100
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_avg = cv_scores.mean() * 100
    results[name] = {"accuracy": score, "cv_score": cv_avg}
    print(f"  {name:<25} → accuracy: {score:.1f}% | cv score: {cv_avg:.1f}%")


print("\n--- Picking best model ---")
best_name = max(results, key=lambda x: results[x]["cv_score"])
best_score = results[best_name]["cv_score"]
print(f"🏆 Best model: {best_name} ({best_score:.1f}% cv score)")


best_model = models[best_name]
best_preds = best_model.predict(X_test)

print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, best_preds,
      target_names=["Low rated", "High rated"]))


cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low rated", "High rated"],
            yticklabels=["Low rated", "High rated"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_movies.png")
plt.show()


print("\n--- Feature Importance (what matters most?) ---")
if best_name in ["Decision Tree", "Random Forest"]:
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(importance.to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.bar(importance["feature"],
            importance["importance"], color="steelblue")
    plt.title(f"Feature Importance — {best_name}")
    plt.ylabel("Importance score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()


print("\n--- Predict a brand new movie ---")
new_movie = pd.DataFrame([[150000000, 500000000, 120, 45.0, 2000]],
            columns=["budget", "revenue", "runtime",
                     "popularity", "vote_count"])

prediction = best_model.predict(new_movie)
probability = best_model.predict_proba(new_movie)[0]

result = "High rated ⭐" if prediction[0] == 1 else "Low rated 👎"
print(f"  Budget:     $150,000,000")
print(f"  Revenue:    $500,000,000")
print(f"  Runtime:    120 mins")
print(f"  Popularity: 45.0")
print(f"  Vote count: 2000")
print(f"  Prediction: {result}")
print(f"  Confidence: {max(probability)*100:.1f}%")


results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")
print(f"\nResults saved to model_results.csv")

print("\n=============================")
print("   PROJECT COMPLETE! ✓      ")
print("=============================")
print("Files saved:")
print("  - rating_distribution.png")
print("  - confusion_matrix_movies.png")
print("  - feature_importance.png")
print("  - model_results.csv")

# Feature importance using Random Forest regardless of winner
print("\n--- Feature Importance (Random Forest) ---")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print(importance.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(importance["feature"], importance["importance"], color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.ylabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("feature_importance.png saved!")