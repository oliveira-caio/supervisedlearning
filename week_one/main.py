import numpy as np


tol = 1e-15
train_set = [[1, 1], [2, 2], [3, 3]]
size = len(train_set)

def f(w, b, x):
    return w*x + b

def cost(w, b):
    return (sum((f(w, b, train_set[i][0]) - train_set[i][1])**2
                for i in range(size)) / 2*size)

def grad_cost(w, b):
    grad_w = (sum((f(w, b, train_set[i][0]) - train_set[i][1])*train_set[i][0]
                  for i in range(size)) / size)
    grad_b = (sum((f(w, b, train_set[i][0]) - train_set[i][1])
                  for i in range(size)) / size)
    return np.array([grad_w, grad_b])

def test(x, y):
    return np.array([2*x, 2*y])

def gradient_descent(f, grad_f, initial, n_iter=1000):
    prev = initial
    for _ in range(n_iter):
        new = prev - 0.35*grad_f(*prev)
        if np.linalg.norm(grad_f(*new)) < tol:
            return new
        prev = new

print(gradient_descent(cost, grad_cost, np.array([0, 0])))
