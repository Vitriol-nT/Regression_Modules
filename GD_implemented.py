#Linear Regression ----------------------------------------------------------------------------------
class LR2:
    def __init__(self, I_val, II_val, iterations, eta):
        self.I_val = list(I_val)
        self.II_val = list(II_val)
        self.iterations = iterations
        self.eta = eta
        self.best_slope = 0
        self.best_intercept = 0
        self.historyQ = []
        self.historyI = []



    def LR2Essence(self):

        n = len(self.I_val)
        w1 = 0  # intercept
        w2 = 0  # slope

        for _ in range(self.iterations):
            grad_w1 = 0
            grad_w2 = 0
            Q = 0

            for i in range(n):
                x = self.I_val[i]
                y_true = self.II_val[i]
                y_pred = w2 * x + w1
                error = y_pred - y_true

                Q += error ** 2
                grad_w1 += 2 * error
                grad_w2 += 2 * error * x

            self.historyQ.append(Q)
            self.historyI.append(_)

            Q /= n
            grad_w1 /= n
            grad_w2 /= n

            w1 -= self.eta * grad_w1
            w2 -= self.eta * grad_w2

        self.best_intercept = w1
        self.best_slope = w2

        print(f"MSE: {Q}")
        print(f"Final model: y = {w2} * x + {w1}")
#Linear Regression ----------------------------------------------------------------------------------

#Poly3 // no GPT used.
#TODO: Make this available according to users input of order, and make it available regardless of how much orders it should be containing.

class poly3:
    def __init__(self, I_val, II_val, iterations, eta):
        self.I_val = list(I_val)
        self.II_val = list(II_val)
        self.iterations = iterations
        self.eta = eta
        self.constI = 0
        self.constII = 0
        self.constIII =0
        self.constIV = 0
        self.historyQ = []
        self.historyI = []

    def Poly3Essence(self):
        n = len(self.I_val)
        w1 = -1
        w2 = -1
        w3 = -1
        w4 = -1

        for _ in range(self.iterations):
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
            grad_w4 = 0
            Q = 0

            for i in range(n):
                x = self.I_val[i]
                y_true = self.II_val[i]
                y_pred = w4 + w1 * x + w2 * x ** 2 + w3 * x ** 3
                error = y_pred - y_true

                Q += error ** 2
                grad_w4 += 2 * error
                grad_w3 += 2 * error * x ** 3
                grad_w2 += 2 * error * x ** 2
                grad_w1 += 2 * error * x

            Q /= n
            grad_w1 /= n
            grad_w2 /= n
            grad_w3 /= n
            grad_w4 /= n

            self.historyQ.append(Q)
            self.historyI.append(_)

            w1 -= self.eta * grad_w1
            w2 -= self.eta * grad_w2
            w3 -= self.eta * grad_w3
            w4 -= self.eta * grad_w4

        self.constI = w1
        self.constII = w2
        self.constIII = w3
        self.constIV = w4

        print(f"MSE: {Q}")
        print(f"Final model: y = {w3}X^3 + {w2}X^2 + {w1}X + {w4}")

    def AdditionalIterations(self, iteration):
        w1 = self.constI
        w2 = self.constII
        w3 = self.constIII
        w4 = self.constIV

        for i in range(iteration):
            n = len(self.I_val)
            Q = 0
            grad_w1 = 0
            grad_w2 = 0
            grad_w3 = 0
            grad_w4 = 0
            for _ in range(n):
                x = self.I_val[_]
                y_true = self.II_val[_]
                y_pred = w4 + w1 * x + w2 * x ** 2 + w3 * x ** 3
                error = y_pred - y_true

                Q += error ** 2
                grad_w4 += 2 * error
                grad_w3 += 2 * error * x ** 3
                grad_w2 += 2 * error * x ** 2
                grad_w1 += 2 * error * x

            Q /= n
            grad_w1 /= n
            grad_w2 /= n
            grad_w3 /= n
            grad_w4 /= n

            self.historyQ.append(Q)
            self.historyI.append(i + self.iterations)

            w1 -= self.eta * grad_w1
            w2 -= self.eta * grad_w2
            w3 -= self.eta * grad_w3
            w4 -= self.eta * grad_w4

        self.constI = w1
        self.constII = w2
        self.constIII = w3
        self.constIV = w4

        print(f"MSE: {Q}")
        print(f"Model after Additional iterations: y = {w3}X^3 + {w2}X^2 + {w1}X + {w4}")

    def predict_x(self, input):
        y = input ** 3 * self.constIII + input ** 2 * self.constII + input ** 3 * self.constI + self.constIV
        print(f"predicted value on {input} = {y}")
        return y
#Poly3
