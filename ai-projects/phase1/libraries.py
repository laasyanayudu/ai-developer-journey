import pandas as pd
import matplotlib.pyplot as plt
data = {
    "model":    ["BERT", "GPT-4", "ResNet", "LSTM", "XGBoost"],
    "accuracy": [94.2,   97.5,    89.1,     85.3,   91.7],
    "speed":    [120,    340,     95,    60,     45]
}
df= pd.DataFrame(data)
print("--- Full dataset ---")
print(df)

print("\n--- Basic Stats ---")
print(df.describe())

print("\n--- Best model ---")
best = df[df["accuracy"] == df["accuracy"].max()]
print(best)

# --- MATPLOTLIB: draw a chart ---

plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["accuracy"], color="steelblue")
plt.title("AI Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy %")
plt.ylim(80, 100)
plt.tight_layout()
plt.savefig("chart.png")
plt.show()

print("\nChart saved as chart.png!")
