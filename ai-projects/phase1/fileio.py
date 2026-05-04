import pandas as pd
import json

# --- PART 1: Reading a CSV file ---
print("=== Reading CSV File ===")

df = pd.read_csv("models.csv")
print(df)

print("\nModels released after 2017:")
recent = df[df["year"] > 2017]
print(recent)

print("\nAverage accuracy:", df["accuracy"].mean())

# --- PART 2: Writing a file ---
print("\n=== Writing a new CSV ===")

# Filter only accurate models and save to new file
good_models = df[df["accuracy"] > 90]
good_models.to_csv("good_models.csv", index=False)
print("Saved good_models.csv with", len(good_models), "models")

# --- PART 3: Working with JSON (how APIs send data) ---
print("\n=== Working with JSON ===")

# This is exactly what an AI API response looks like
api_response = '''
{
    "model": "claude-3",
    "response": "The answer is 42",
    "tokens_used": 150,
    "confidence": 0.97
}
'''

data = json.loads(api_response)
print("Model used:", data["model"])
print("Response:", data["response"])
print("Confidence:", data["confidence"])

# --- PART 4: Saving results as JSON ---
results = {
    "best_model": "GPT-4",
    "accuracy": 97.5,
    "tested_models": 5
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved results.json!")
print("Done — you can now read and write real data files!")