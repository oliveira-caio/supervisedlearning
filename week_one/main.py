import numpy as np


tol = 1e-15
dataset = [[1, 1], [2, 2], [3, 3]]
size = len(dataset)

def f(w, b, x):
    return w*x + b

def grad_cost(w, b):
    grad_w = (sum((f(w, b, x) - y)*x for x, y in dataset) / size)
    grad_b = (sum((f(w, b, x) - y) for x, y in dataset) / size)
    return np.array([grad_w, grad_b])

def gradient_descent(grad_F, initial, n_iter=1000, learning_rate=0.35):
    prev = initial
    for i in range(n_iter):
        new = prev - learning_rate*grad_F(*prev)
        if np.linalg.norm(grad_F(*new)) < tol:
            print(f'Converged to ({new[0]}, {new[1]}) in {i} iterations.')
            return
        prev = new
    raise Exception('Didn\'t converge.')

def main():
    gradient_descent(grad_cost, np.array([0, 0]))

if __name__ == '__main__':
    main()
