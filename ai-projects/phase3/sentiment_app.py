from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================
# PHASE 3 PROJECT: SENTIMENT ANALYSIS APP
# By: Chetana Laasya Nayudu
# ============================================

print("="*50)
print("   SENTIMENT ANALYSIS APP")
print("="*50)


print("\n--- Step 1: Loading data ---")
df = pd.read_csv("reviews.csv")
print(f"Loaded {len(df)} reviews across {df['category'].nunique()} categories")
print(f"Categories: {list(df['category'].unique())}")
print(f"\nSample reviews:")
print(df[["product", "category", "review"]].head(3).to_string(index=False))


print("\n--- Step 2: Loading BERT sentiment model ---")
print("Loading... (using cached model, should be fast!)")
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded!")

print("\n--- Step 3: Analyzing reviews ---")
labels = []
scores = []

for i, review in enumerate(df["review"]):
    result = sentiment(review)[0]
    labels.append(result["label"])
    scores.append(round(result["score"] * 100, 1))
    print(f"  [{i+1:>2}/{len(df)}] {result['label']:<8} "
          f"{result['score']*100:.1f}% | {review[:40]}...")

df["sentiment"] = labels
df["confidence"] = scores


print("\n--- Step 4: Overall Summary ---")
total = len(df)
positive = (df["sentiment"] == "POSITIVE").sum()
negative = (df["sentiment"] == "NEGATIVE").sum()
avg_confidence = df["confidence"].mean()

print(f"Total reviews:    {total}")
print(f"Positive:         {positive} ({positive/total*100:.1f}%)")
print(f"Negative:         {negative} ({negative/total*100:.1f}%)")
print(f"Avg confidence:   {avg_confidence:.1f}%")


print("\n--- Step 5: Category Breakdown ---")
category_stats = []

for category in df["category"].unique():
    cat_df = df[df["category"] == category]
    pos = (cat_df["sentiment"] == "POSITIVE").sum()
    neg = (cat_df["sentiment"] == "NEGATIVE").sum()
    avg_conf = cat_df["confidence"].mean()
    pos_pct = pos / len(cat_df) * 100

    category_stats.append({
        "category": category,
        "total": len(cat_df),
        "positive": pos,
        "negative": neg,
        "positive_%": round(pos_pct, 1),
        "avg_confidence": round(avg_conf, 1)
    })

    print(f"\n  {category}:")
    print(f"    Positive: {pos}/{len(cat_df)} ({pos_pct:.1f}%)")
    print(f"    Negative: {neg}/{len(cat_df)}")
    print(f"    Avg confidence: {avg_conf:.1f}%")

stats_df = pd.DataFrame(category_stats)


print("\n--- Step 6: Insights ---")
best_category = stats_df.loc[stats_df["positive_%"].idxmax(), "category"]
worst_category = stats_df.loc[stats_df["positive_%"].idxmin(), "category"]
best_review_idx = df["confidence"][df["sentiment"] == "POSITIVE"].idxmax()
worst_review_idx = df["confidence"][df["sentiment"] == "NEGATIVE"].idxmax()

print(f"🏆 Most positive category: {best_category}")
print(f"⚠️  Most negative category: {worst_category}")
print(f"\n✅ Most positive review:")
print(f"   {df.loc[best_review_idx, 'product']}: "
      f"{df.loc[best_review_idx, 'review']}")
print(f"\n❌ Most negative review:")
print(f"   {df.loc[worst_review_idx, 'product']}: "
      f"{df.loc[worst_review_idx, 'review']}")


print("\n--- Step 7: Creating charts ---")
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Sentiment Analysis Report", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

# Chart 1 — Overall pie chart
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie(
    [positive, negative],
    labels=["Positive", "Negative"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.1f%%",
    startangle=90
)
ax1.set_title("Overall Sentiment")

# Chart 2 — Category bar chart
ax2 = fig.add_subplot(gs[0, 1])
categories = stats_df["category"]
x = np.arange(len(categories))
width = 0.35
ax2.bar(x - width/2, stats_df["positive"],
        width, label="Positive", color="#2ecc71")
ax2.bar(x + width/2, stats_df["negative"],
        width, label="Negative", color="#e74c3c")
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_title("Sentiment by Category")
ax2.set_ylabel("Count")
ax2.legend()

# Chart 3 — Confidence by category
ax3 = fig.add_subplot(gs[1, 0])
colors = ["#2ecc71" if p >= 60 else "#e74c3c"
          for p in stats_df["positive_%"]]
ax3.bar(categories, stats_df["positive_%"], color=colors)
ax3.axhline(y=60, color="gray", linestyle="--", label="60% threshold")
ax3.set_title("Positive % by Category")
ax3.set_ylabel("Positive %")
ax3.set_ylim(0, 100)
ax3.legend()

# Chart 4 — Individual review confidence
ax4 = fig.add_subplot(gs[1, 1])
colors_individual = ["#2ecc71" if s == "POSITIVE" else "#e74c3c"
                     for s in df["sentiment"]]
ax4.bar(range(len(df)), df["confidence"], color=colors_individual)
ax4.set_title("Confidence per Review")
ax4.set_xlabel("Review #")
ax4.set_ylabel("Confidence %")
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("sentiment_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as sentiment_report.png!")


print("\n--- Step 8: Saving results ---")
df.to_csv("sentiment_results.csv", index=False)
stats_df.to_csv("category_stats.csv", index=False)
print("Saved sentiment_results.csv")
print("Saved category_stats.csv")


print("\n" + "="*50)
print("   FINAL REPORT")
print("="*50)
print(f"\nAnalyzed {total} product reviews using BERT")
print(f"Overall sentiment: "
      f"{positive/total*100:.1f}% positive")
print(f"Average model confidence: {avg_confidence:.1f}%")
print(f"\nCategory rankings (best to worst):")
ranked = stats_df.sort_values("positive_%", ascending=False)
for i, row in ranked.iterrows():
    bar = "█" * int(row["positive_%"] / 10)
    print(f"  {row['category']:<15} "
          f"{bar:<10} {row['positive_%']}%")

print("\nFiles saved:")
print("  - sentiment_report.png")
print("  - sentiment_results.csv")
print("  - category_stats.csv")
print("\n✓ Phase 3 Project Complete!")