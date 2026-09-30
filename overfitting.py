import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# random data, a curve
X = np.linspace(-3, 3, 30)

y = 2 * X**2 + X + 3 + np.random.normal(0, 2, size=len(X))

X = X.reshape(-1, 1)

# plt.scatter(X, y)
# plt.xlabel("X")
# plt.ylabel("y")
# plt.title("Our synthetic dataset")
# plt.show()

# print(X)
# print(y)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model = LinearRegression()
# model.fit(X_train, y_train)

# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)

# print("Train MSE:", mean_squared_error(y_train, train_pred))
# print("Test MSE:", mean_squared_error(y_test, test_pred))

# baseline comparision
from sklearn.metrics import r2_score

# baseline_prediction = y_train.mean()

# baseline = np.full(
#     shape=len(y_test),
#     fill_value=baseline_prediction
# )

# baseline_mse = mean_squared_error(y_test, baseline)
# baseline_r2 = r2_score(y_test, baseline)

# print("Baseline prediction:", baseline_prediction)
# print("Baseline MSE:", baseline_mse)
# print("Baseline R²:", baseline_r2)

# print("Model MSE:", mean_squared_error(y_test, test_pred))
# print("Model R²:", r2_score(y_test, test_pred))


# giving it "powered" data
# from sklearn.preprocessing import PolynomialFeatures
# degrees = [2, 5, 10 , 20]
# for degree in degrees:
#     poly = PolynomialFeatures(degree=degree, include_bias=False)

#     X_train_poly = poly.fit_transform(X_train)
#     X_test_poly = poly.transform(X_test)

#     model_poly = LinearRegression()
#     model_poly.fit(X_train_poly, y_train)

#     train_pred_poly = model_poly.predict(X_train_poly)
#     test_pred_poly = model_poly.predict(X_test_poly)

#     print(f"\nPolynomial degree {degree}:")

#     print(
#         "   Train MSE:",
#         mean_squared_error(y_train, train_pred_poly)
#     )

#     print(
#         "   Test MSE:",
#         mean_squared_error(y_test, test_pred_poly)
#     )

#     print(
#         "   Test R²:",
#         r2_score(y_test, test_pred_poly)
#     )

# plot diffrent degree predictions
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# degrees = [1, 2, 5, 10]

# x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# plt.scatter(X, y)

# for degree in degrees:
#     poly = PolynomialFeatures(
#         degree=degree,
#         include_bias=False
#     )

#     X_poly = poly.fit_transform(X)

#     model = LinearRegression()
#     model.fit(X_poly, y)

#     x_plot_poly = poly.transform(x_plot)
#     y_plot = model.predict(x_plot_poly)

#     plt.plot(
#         x_plot,
#         y_plot,
#         label=f"Degree {degree}"
#     )

# plt.legend()
# plt.xlabel("X")
# plt.ylabel("y")
# plt.show()

# regularization
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=10, include_bias=False)

# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)
# alphas = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# for alpha in alphas:
#     model = Ridge(alpha=alpha)
    
#     model.fit(X_train_poly, y_train)
    
#     train_pred = model.predict(X_train_poly)
#     test_pred = model.predict(X_test_poly)
    
#     print(f"\nAlpha: {alpha}")
#     print("Train MSE:", mean_squared_error(y_train, train_pred))
#     print("Test MSE:", mean_squared_error(y_test, test_pred))
#     print("Test R²:", r2_score(y_test, test_pred))

# cross validation
# from sklearn.linear_model import RidgeCV

# alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

# ridge_cv = RidgeCV(
#     alphas=alphas,
#     cv=5
# )

# ridge_cv.fit(X_train_poly, y_train)

# print("Best alpha:", ridge_cv.alpha_)

# test_pred = ridge_cv.predict(X_test_poly)

# print("Test MSE:", mean_squared_error(y_test, test_pred))
# print("Test R²:", r2_score(y_test, test_pred))

# pipeline with polynomial regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]

model = Pipeline([
    ("poly", PolynomialFeatures(degree=10, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=5))
])

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Best alpha:", model.named_steps["ridge"].alpha_)

print("Train MSE:", mean_squared_error(y_train, train_pred))
print("Test MSE:", mean_squared_error(y_test, test_pred))

print("Test R²:", r2_score(y_test, test_pred))