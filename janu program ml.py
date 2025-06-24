import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([11,12,13,14,15,16,17,18,19]).reshape(-1,1)
y = np.array([11,13,12,15,17,18,18,19,20])

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

print(f"slope(m): {model.coef_[0]:.2f}")
print(f"intercept(b): {model.intercept_:.2f}")

plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Simple Linear Regression")
plt.show()
