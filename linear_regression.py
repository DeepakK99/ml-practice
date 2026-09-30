import numpy as np

# training data
X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([35, 42, 51, 58, 67, 73], dtype=float)

# parameters
w = 5.0
b = 30.0

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):
    # 1. make predictions
    predictions = w * X + b

    # 2. calculate errors
    errors = y - predictions

    # 3. calculate errors
    dw = (-2 / n) * np.sum(X * errors)
    db = (-2 / n) * np.sum(errors)

    # 4. update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 5. make predictions
    predictions = w * X + b
    
    # 6. calculate errors
    errors = y - predictions

    # calculate loss
    mse = np.mean(errors ** 2)


    if epoch % 100 == 0:
        print(
            f"epoch={epoch}, "
            f"w={w:.4f}, "
            f"b={b:.4f}, "
            f"mse={mse:.4f}, "
        )

print("\nFinal:")
print("w =", w)
print("b =", b)

print("Prediction for X = y: ", (w * 7) + b)