from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ============================================
# CONCEPT 4: FINE-TUNING
# Teaching a pre-trained model new tricks
# ============================================

print("=============================")
print("   FINE-TUNING DEMO         ")
print("=============================")


print("\n--- Part 1: How models read text ---")
print("Models don't read words — they read TOKENS (numbers)")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

sentences = [
    "I love machine learning",
    "AI is changing the world",
    "Python is great for data science"
]

print("\nTokenizing sentences:")
for sentence in sentences:
    tokens = tokenizer.tokenize(sentence)
    ids    = tokenizer.encode(sentence)
    print(f"\n  Text:   {sentence}")
    print(f"  Tokens: {tokens}")
    print(f"  IDs:    {ids}")


print("\n--- Part 2: Pre-trained model ---")
print("Using distilbert already downloaded (no new download!)")

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


ai_reviews = [
    "This AI tool saved me hours of work every day",
    "The model keeps making wrong predictions",
    "Incredible accuracy on our dataset",
    "Too slow for production use",
    "Best ML library I have ever used",
]

print("\nAI tool reviews — before fine-tuning:")
for review in ai_reviews:
    result = sentiment(review)[0]
    label = result["label"]
    score = result["score"] * 100
    emoji = "😊" if label == "POSITIVE" else "😞"
    print(f"  {emoji} {score:.1f}% | {review[:45]}...")


print("\n--- Part 3: Custom training data ---")
print("In real fine-tuning you would train on YOUR labeled data")


custom_data = {
    "text": [
        "The API response time is excellent",
        "Model accuracy dropped after update",
        "Integration was seamless and easy",
        "Too many false positives in results",
        "Training completed faster than expected",
        "The documentation is confusing",
        "Predictions are consistently accurate",
        "System crashed during inference",
        "Easy to deploy and scale",
        "Poor performance on edge cases",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
   
}

df = pd.DataFrame(custom_data)
print(f"\nCustom dataset: {len(df)} labeled examples")
print(df.to_string(index=False))


print("\n--- Part 4: Feature Extraction ---")
print("Using BERT as a feature extractor + simple classifier")
print("This is the lightweight version of fine-tuning")

# Extract features using the pre-trained model
feature_extractor = pipeline(
    "feature-extraction",
    model="distilbert-base-uncased",
    return_tensor=False
)

print("\nExtracting features from custom data...")
features = []
for text in df["text"]:
    
    embedding = feature_extractor(text)[0][0]  
    features.append(embedding)

X = np.array(features)
y = np.array(df["label"])

print(f"Feature shape: {X.shape}")
print(f"Each text → {X.shape[1]} numbers that capture its meaning")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X, y, cv=3)

print(f"\nClassifier accuracy: {scores.mean()*100:.1f}%")
print("(Trained on only 10 examples using BERT features!)")


print("\n--- Part 5: Predicting new AI reviews ---")
clf.fit(X, y)

new_reviews = [
    "The inference speed is impressive",
    "Model failed to converge during training",
    "Beautiful API design and documentation",
]

print("\nNew review predictions:")
for review in new_reviews:
    embedding = feature_extractor(review)[0][0]
    pred = clf.predict([embedding])[0]
    prob = clf.predict_proba([embedding])[0]
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    confidence = max(prob) * 100
    print(f"\n  Review: {review}")
    print(f"  Result: {label} ({confidence:.1f}% confident)")

print("\n=============================")
print("   FINE-TUNING COMPLETE ✓   ")
print("=============================")
print("\nWhat you learned:")
print("  1. Models read tokens (numbers), not words")
print("  2. Pre-trained models work great out of the box")
print("  3. Fine-tuning = pre-trained features + your data")
print("  4. Even 10 examples can give good results with BERT!")