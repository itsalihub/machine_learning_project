
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data = load_diabetes()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Средноквадратична грешка (MSE):", mse)
print("Коефициент на детерминация (R²):", r2)


plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=data.feature_names, rounded=True)
plt.title("Decision Tree (max_depth=3)", fontsize=16)
plt.show()
