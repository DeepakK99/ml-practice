# sample tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# X = [[20], [25], [30], [35], [40], [45]]
# y = [0, 1, 0, 1, 1, 0]

# tree = DecisionTreeClassifier(
#     criterion="gini",
#     max_depth=2,
#     random_state=42
# )

# tree.fit(X, y)

# plt.figure(figsize=(12, 6))

# plot_tree(
#     tree,
#     feature_names=["Age"],
#     class_names=["No", "Yes"],
#     filled=True
# )

# plt.show()

# using breast cancer data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

depths = [1, 2, 3, 5, 10, None]

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Depth: {depth}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print()

    if depth == 3:
        for name, importance in zip(data.feature_names, model.feature_importances_):
            print(f"{name}: {importance:.4f}")

