from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([
    [20, 30000],
    [30, 50000],
    [40, 100000],
    [50, 200000],
    [60, 400000]
])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original:")
print(X)

print("\nScaled:")
print(X_scaled)

print("\nMeans:")
print(X_scaled.mean(axis=0))

print("\nStandard deviations:")
print(X_scaled.std(axis=0))