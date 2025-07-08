from LR2_implemented_GD import *
import matplotlib.pyplot as plt
import numpy as np

one = [
    -3.46,
    -1.78,
    -2.26,
    -1,
    -1.45,
    -1.5,
    -0.36,
    0.86,
    0.6,
    1.8
]
two = [
    2.98,
    2.24,
    1.08,
    1,
    1.28,
    0.58,
    0.5,
    0.72,
    -0.78,
    -1.6
]

Example = LR2(one, two, 100, 0.01)
Example.LR2Essence()

x = np.linspace(min(one), max(one), 1000 )


plt.scatter(one, two)
a = Example.best_slope
b = Example.best_intercept
y = a * x + b
plt.plot(x, y)
plt.title("Regression Status")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

h = Example.historyQ
i = Example.historyI
plt.scatter(i, h)
plt.plot(i, h)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Model Training Error")
plt.show()
