from transformers import pipeline
import pandas as pd

# ============================================
# CONCEPT 3: HUGGING FACE & NLP
# Using pre-trained models for text analysis
# ============================================

print("=============================")
print("   NLP WITH HUGGING FACE    ")
print("=============================")


print("\n--- Part 1: Sentiment Analysis ---")
print("Loading model... (first time downloads ~500MB, be patient!)")

sentiment = pipeline("sentiment-analysis")

reviews = [
    "This movie was absolutely amazing, I loved every minute!",
    "Terrible film, complete waste of time and money.",
    "The acting was okay but the story was boring.",
    "One of the best experiences I have ever had!",
    "I would not recommend this to anyone.",
]

print("\nAnalyzing reviews:")
for review in reviews:
    result = sentiment(review)[0]
    label = result["label"]
    score = result["score"] * 100
    emoji = "😊" if label == "POSITIVE" else "😞"
    print(f"\n  Text: {review[:50]}...")
    print(f"  {emoji} {label} — confidence: {score:.1f}%")


print("\n--- Part 2: Zero-shot Classification ---")
print("Loading model...")

classifier = pipeline("zero-shot-classification")

texts = [
    "The stock market crashed today losing 500 points",
    "Scientists discover new species of dinosaur in Argentina",
    "The team won the championship after an amazing comeback",
]

categories = ["finance", "science", "sports", "politics"]

print("\nClassifying texts:")
for text in texts:
    result = classifier(text, candidate_labels=categories)
    top_label = result["labels"][0]
    top_score = result["scores"][0] * 100
    print(f"\n  Text: {text[:55]}...")
    print(f"  Category: {top_label} ({top_score:.1f}% confident)")


print("\n--- Part 3: Text Generation ---")
print("Loading model...")

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence will change the world by"
print(f"\nPrompt: {prompt}")

result = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1,
    truncation=True
)

print(f"Generated: {result[0]['generated_text']}")


print("\n--- Part 4: Named Entity Recognition ---")
print("Loading model...")

ner = pipeline("ner", aggregation_strategy="simple")

text = "Elon Musk founded SpaceX in California and Tesla in Silicon Valley."
print(f"\nText: {text}")
print("\nEntities found:")

entities = ner(text)
for entity in entities:
    print(f"  {entity['word']:<20} → {entity['entity_group']}")

print("\n=============================")
print("   NLP COMPLETE! ✓          ")
print("=============================")