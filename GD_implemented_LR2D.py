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
