import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# z = np.array([-100, -10, -1, 0, 1, 10, 100])

# print(sigmoid(z))

# X = [0, 1, 1.5, 2, 3]
# w = 2
# b = -3

# print([sigmoid((w * x) + b) for x in X])


def log_loss_single(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

print("y=1, p=0.9:", log_loss_single(1, 0.9))
print("y=1, p=0.1:", log_loss_single(1, 0.1))

print("y=0, p=0.1:", log_loss_single(0, 0.1))
print("y=0, p=0.9:", log_loss_single(0, 0.9))
