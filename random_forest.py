from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

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

# experimenting with different configs
configs = {"max_depth": [2, 3, 5, 10, None],
           "min_samples_leaf": [1, 2, 5, 10],
           "max_features": ["sqrt", "log2", None],
           }

for config_key in configs:
    for config_value in configs[config_key]:
        current_config = {config_key: config_value}
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            **current_config
        )

        rf.fit(X_train_scaled, y_train)

        y_train_pred = rf.predict(X_train_scaled)
        y_test_pred = rf.predict(X_test_scaled)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        print(f"{config_key}: {config_value}")
        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

        print("Precision:", precision_score(y_test, y_test_pred))
        print("Recall:", recall_score(y_test, y_test_pred))
        print("F1:", f1_score(y_test, y_test_pred))
        print()
    print()