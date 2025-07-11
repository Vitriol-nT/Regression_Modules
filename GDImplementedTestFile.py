import math

from GD_implemented import *
import matplotlib.pyplot as plt
import numpy as np

import random

w3_true = 10
w2_true = 0.25

one = [random.uniform(-10, 10) for _ in range(100)]
two = [w3_true * math.sin(w2_true * x) + random.gauss(0, 2.5) for x in one]

Example = poly3(one, two, 10000, 0.000003)
Example.Poly3Essence()
for i in range(2):
    print(f"Additional Iteration: {i}")
    Example.AdditionalIterations(100000)

Example.predict_x(3)

plt.scatter(one, two)
a = Example.constI
b = Example.constII
c = Example.constIII
d = Example.constIV

x = np.linspace(min(one), max(one), 1000)

y = c * x ** 3 + b * x ** 2 + a * x + b
plt.scatter(one, two)
plt.plot(x, y)
plt.show()

h = Example.historyQ
i = Example.historyI
plt.scatter(i, h)
plt.plot(i, h)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Model Training Error")
plt.show()

