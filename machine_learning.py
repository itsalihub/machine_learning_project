
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
y_tree_pred = tree_model.predict(X_test)

plt.figure(figsize=(20, 10))
plt.figure(figsize=(14, 8))  # Увеличаваме размера на графиката
plot_tree(tree_model, filled=True, feature_names=diabetes.feature_names, fontsize=10)
plt.title("По-компактно дърво на решенията (max_depth=3)")
plt.tight_layout()
plt.show()
plt.title("Дърво на решенията за прогнозиране на диабет")
plt.show()

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

mse_tree = mean_squared_error(y_test, y_tree_pred)
r2_tree = r2_score(y_test, y_tree_pred)

mse_rf = mean_squared_error(y_test, y_rf_pred)
r2_rf = r2_score(y_test, y_rf_pred)

print("=== Оценка на Decision Tree ===")
print(f"MSE: {mse_tree:.2f}")
print(f"R² Score: {r2_tree:.2f}\n")

print("=== Оценка на Random Forest ===")
print(f"MSE: {mse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")

sample_patient = X_test[0].reshape(1, -1)
tree_prediction = tree_model.predict(sample_patient)
rf_prediction = rf_model.predict(sample_patient)

print("\nПрогнозирана стойност за примерен пациент:")
print(f"Decision Tree: {tree_prediction[0]:.2f}")
print(f"Random Forest: {rf_prediction[0]:.2f}")
