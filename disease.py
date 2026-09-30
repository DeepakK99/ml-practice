# load the dataset
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

# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# logistic regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

model.fit(X_train_scaled, y_train)

# print("Weights shape:", model.coef_.shape)
# print("Bias shape:", model.intercept_.shape)

# print("\nWeights:")
# print(model.coef_)

# print("\nBias:")
# print(model.intercept_)

# probabilities and threshold behaviour
# y_proba = model.predict_proba(X_test_scaled)

# print(y_proba[:5])
# print("Shape:", y_proba.shape)

# y_pred = model.predict(X_test_scaled)

# print(y_pred[:20])

# y_proba = model.predict_proba(X_test_scaled)[:, 1]

# print("First 10 probabilities:")
# print(y_proba[:10])

# print("\nFirst 10 predictions:")
# print(model.predict(X_test_scaled)[:10])

# for threshold in [0.3, 0.5, 0.8]:
#     y_pred_custom = (y_proba >= threshold).astype(int)
    
#     print("\nThreshold:", threshold)
#     print("Predicted positives:", y_pred_custom.sum())

# metrices
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# y_pred = model.predict(X_test_scaled)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# k fold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42, max_iter=1000))
])

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(
#     pipeline,
#     X_train,
#     y_train,
#     cv=cv,
#     scoring="accuracy"
# )

# print("Fold accuracies:", scores)
# print("Mean accuracy:", scores.mean())
# print("Std accuracy:", scores.std())

from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# print("Best C:", grid_search.best_params_["model__C"])
# print("Best CV accuracy:", grid_search.best_score_)

# results = grid_search.cv_results_

# for mean, std, params in zip(
#     results["mean_test_score"],
#     results["std_test_score"],
#     results["params"]
# ):
#     print(
#         params,
#         "mean =", mean,
#         "std =", std
#     )

best_model = grid_search.best_estimator_

# y_test_pred = best_model.predict(X_test)

# print("Test accuracy:", accuracy_score(y_test, y_test_pred))
# print("Test precision:", precision_score(y_test, y_test_pred))
# print("Test recall:", recall_score(y_test, y_test_pred))
# print("Test F1:", f1_score(y_test, y_test_pred))

# print("Best parameters:", grid_search.best_params_)
# print("Best CV score:", grid_search.best_score_)

# from sklearn.metrics import roc_auc_score

y_test_proba = best_model.predict_proba(X_test)[:, 1]

# auc = roc_auc_score(y_test, y_test_proba)

# print("ROC-AUC:", auc)

# plot the roc-auc curve
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

# fpr, tpr, thresholds = roc_curve(
#     y_test,
#     y_test_proba
# )

# plt.plot(fpr, tpr)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.show()

# plot precision recall curve
# y_test_proba = best_model.predict_proba(X_test)[:, 1]
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt

# precision, recall, thresholds = precision_recall_curve(
#     y_test,
#     y_test_proba
# )

# plt.plot(recall, precision)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.show()

# from sklearn.metrics import average_precision_score

# ap = average_precision_score(y_test, y_test_proba)

# print("Average Precision:", ap)

# playing with threshold
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds_to_test:
    y_pred = (y_test_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(
        f"Threshold={threshold:.1f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

