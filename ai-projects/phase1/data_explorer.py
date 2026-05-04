import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# DATA EXPLORER — Phase 1 Portfolio Project
# By: Chetana Laasya nayudu
# ============================================

def load_data(filename):
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def summarize(df):
   
    print("\n--- Dataset Summary ---")
    print(df.describe().round(2))

   
    best = df[df["accuracy"] == df["accuracy"].max()]
    print(best.to_string(index=False))

    
    print(round(df["accuracy"].mean(), 2), "%")

def plot_chart(df):
   
    plt.figure(figsize=(9, 5))

    colors = ["green" if a > 90 else "steelblue" for a in df["accuracy"]]
    bars = plt.bar(df["model"], df["accuracy"], color=colors)

   
    for bar, val in zip(bars, df["accuracy"]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{val}%", ha="center", fontsize=10)

    plt.title("AI Model Accuracy Comparison", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Accuracy %")
    plt.ylim(80, 100)
    plt.axhline(y=90, color="red", linestyle="--", label="90% threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_chart.png")
    print("\nChart saved as results_chart.png")
    plt.show()

def save_results(df):
    """Save filtered results to a new CSV"""
    top = df[df["accuracy"] > 90]
    top.to_csv("top_models.csv", index=False)
    print(f"Saved {len(top)} top models to top_models.csv")

# ============================================
# MAIN PROGRAM — runs everything in order
# ============================================

print("=============================")
print("   AI MODEL DATA EXPLORER   ")
print("=============================")

df = load_data("models.csv")
summarize(df)
plot_chart(df)
save_results(df)

print("\nDone! Check your folder for:")
print("  - results_chart.png")
print("  - top_models.csv")