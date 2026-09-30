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

from xgboost import XGBClassifier

for learning_rate in [0.01, 0.05, 0.1, 0.3]:
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=learning_rate,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    print("\nLearning Rate:", learning_rate)
    print("Train Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy:", model.score(X_test, y_test))