import torch
import torch.nn as nn
import numpy as np

# ============================================
# CONCEPT 1: PYTORCH TENSORS
# ============================================

print("=============================")
print("   DEEP LEARNING — PHASE 3  ")
print("=============================")

print("\n--- Tensors ---")


scalar = torch.tensor(7)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]])

print(f"Scalar:      {scalar}          shape: {scalar.shape}")
print(f"Vector:      {vector}   shape: {vector.shape}")
print(f"Matrix:\n{matrix}  shape: {matrix.shape}")


print("\n--- Tensor Operations ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b  = {a + b}")
print(f"a * b  = {a * b}")
print(f"a mean = {a.mean()}")
print(f"a sum  = {a.sum()}")


print("\n--- Random Tensors ---")
random_tensor = torch.rand(3, 4)   
zeros_tensor  = torch.zeros(2, 3)  
ones_tensor   = torch.ones(2, 3)   

print(f"Random tensor (3x4):\n{random_tensor}")
print(f"\nZeros tensor (2x3):\n{zeros_tensor}")
print(f"\nOnes tensor (2x3):\n{ones_tensor}")


print("\n--- Hardware ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("(No GPU detected — using CPU. Fine for learning!)")
else:
    print("(GPU detected — training will be fast!)")


print("\n--- NumPy ↔ PyTorch ---")
numpy_array = np.array([1.0, 2.0, 3.0])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

print(f"NumPy array:   {numpy_array}")
print(f"PyTorch tensor: {torch_tensor}")
print(f"Back to NumPy: {back_to_numpy}")
print("Conversion works both ways!")
# ============================================
# CONCEPT 2: BUILDING A NEURAL NETWORK
# ============================================

print("\n=============================")
print("   NEURAL NETWORK           ")
print("=============================")

# --- Step 1: Create sample data ---
# Predicting if a model is good based on 3 features
# Features: [accuracy, speed, size]
# Label: 1 = good model, 0 = not good

torch.manual_seed(42) 

X = torch.tensor([
    [0.95, 0.80, 0.30],  # high accuracy, fast, small  → good
    [0.72, 0.60, 0.80],  # low accuracy, medium, large → not good
    [0.91, 0.75, 0.40],  # high accuracy, fast, small  → good
    [0.65, 0.50, 0.90],  # low accuracy, slow, large   → not good
    [0.88, 0.70, 0.35],  # high accuracy, fast, small  → good
    [0.70, 0.55, 0.85],  # low accuracy, slow, large   → not good
], dtype=torch.float32)

y = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)

print(f"Input shape:  {X.shape}")   # 6 samples, 3 features
print(f"Output shape: {y.shape}")   # 6 labels


class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()
        # Layer 1: 3 inputs → 8 neurons
        self.layer1 = nn.Linear(3, 8)
        # Layer 2: 8 neurons → 4 neurons
        self.layer2 = nn.Linear(8, 4)
        # Output layer: 4 neurons → 1 output
        self.output = nn.Linear(4, 1)
        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))   
        x = self.relu(self.layer2(x))  
        x = self.sigmoid(self.output(x)) 
        return x


model = SimpleNN()
print(f"\nNeural network architecture:")
print(model)


total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable parameters: {total_params}")


loss_fn = nn.BCELoss()                           
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


print("\n--- Training ---")
epochs = 100

for epoch in range(epochs):
   
    y_pred = model(X).squeeze()

    loss = loss_fn(y_pred, y)

   
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # calculate new gradients
    optimizer.step()        # update weights

   
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:>3}/100 | Loss: {loss.item():.4f}")


print("\n--- Results after training ---")
model.eval() 
with torch.no_grad():  
    predictions = model(X).squeeze()
    predicted_labels = (predictions > 0.5).float()

    correct = (predicted_labels == y).sum().item()
    accuracy = correct / len(y) * 100

    print(f"Accuracy: {accuracy:.1f}%")
    print(f"\nSample by sample:")
    for i in range(len(y)):
        actual = "Good" if y[i] == 1 else "Not good"
        predicted = "Good" if predicted_labels[i] == 1 else "Not good"
        confidence = predictions[i].item() * 100
        match = "✓" if predicted_labels[i] == y[i] else "✗"
        print(f"  {match} Actual: {actual:<10} "
              f"Predicted: {predicted:<10} "
              f"Confidence: {confidence:.1f}%")