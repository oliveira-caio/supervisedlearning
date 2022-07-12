repository for the codes of the [supervised machine learning: regression and classification](https://www.coursera.org/learn/machine-learning?specialization=machine-learning-introduction) course of Andrew Ng.

# first week

a very basic algorithm for linear regression using the gradient descent algorithm to learn. what that means is, if we have a labeled dataset ${(x_1, y_1), \ldots, (x_n, y_n)}$, then this algorithm fits the best linear approximation to the points. The following image gives an example:
![linear regression](./linreg.png)

the algorithm works like this: a linear function can be defined as $f(x) = wx + b$, being $w$ and $b$ the parameters we want to learn to best approximate our dataset.
