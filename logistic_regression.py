import numpy as np

# Data
X = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([0, 0, 0, 1, 1], dtype=float)

# Parameters
w = 0.0
b = 0.0

learning_rate = 0.1
epochs = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for epoch in range(epochs):

    # Forward pass
    z = w * X + b
    p = sigmoid(z)

    # Loss
    loss = -np.mean(
        y * np.log(p) +
        (1 - y) * np.log(1 - p)
    )

    # Gradients
    dw = np.mean((p - y) * X)
    db = np.mean(p - y)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 200 == 0:
        print(
            f"epoch={epoch}, "
            f"w={w:.4f}, "
            f"b={b:.4f}, "
            f"loss={loss:.4f}"
        )

print("\nFinal:")
print("w =", w)
print("b =", b)

z = w * X + b
p = sigmoid(z)

print("\nPredicted probabilities:")
print(p)

print("\nPredicted classes:")
print((p >= 0.5).astype(int))

print("\nActual classes:")
print(y.astype(int))

print("\nDecision boundary:")
print(-b / w)


# experiment thersholds
X_new = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
z_new = w * X_new + b
p_new = sigmoid(z_new)

for threshold in [0.3, 0.5, 0.8]:
    predictions = (p_new >= threshold).astype(int)
    print(threshold, predictions)

# confusion matrix idea
actual = np.array([1, 1, 1, 1, 0, 0, 0, 0])

predicted = np.array([1, 1, 0, 1, 1, 0, 0, 0])

print("Actual:   ", actual)
print("Predicted:", predicted)

# f1 score idea:
cases = [
    (1.0, 1.0),
    (1.0, 0.5),
    (0.5, 1.0),
    (0.9, 0.1),
    (0.1, 0.9),
    (0.5, 0.5),
]
def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

print([f1(precision, recall) for precision, recall in cases])
