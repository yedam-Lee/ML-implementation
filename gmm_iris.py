import numpy as np
import pandas as pd
import pylab as plt
from sklearn import datasets
from collections import namedtuple
from matplotlib.patches import Ellipse


class GMM:

    def __init__(self, k=3, eps=0.0001):
        self.k = k  # the number of clusters
        self.eps = eps  # the threshold to stop `epsilon`

    def EM(self, X, max_iters=1000):

        n, d = X.shape

        # Choose the sample mean randomly
        mu = X[np.random.choice(n, self.k, False), :]

        # Initialize the covariance matrices
        sigma = [np.eye(d)] * self.k

        # Initialize the weights
        w = [1. / self.k] * self.k

        # Assign the class randomly
        y_hat = np.random.choice(3, len(X)) + 1

        # Initialize responsibility matrix as all zeros
        M = np.zeros((n, self.k))

        # The probability density function of each sample
        def func_prob(mu, s):
            prob = np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) * np.exp(
                -.5 * np.einsum('ij,ij->i', X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))
            return prob

        # Repeat up to max_iters iterations
        log_likelihoods = []

        while len(log_likelihoods) < max_iters:

            ### Expectation - Step

            # Compute the log likelihood with the fixed paramaters
            for k in range(self.k):
                M[:, k] = w[k] * func_prob(mu[k], sigma[k])
            log_likelihood = np.sum(np.log(np.sum(M, axis=1)))
            log_likelihoods.append(log_likelihood)
            M = (M.T / np.sum(M, axis=1)).T

            # update the class
            for i in range(X.shape[0]):
                if M[i, 0] >= M[i, 1] and M[i, 0] >= M[i, 2]:
                    y_hat[i] = 1
                elif M[i, 1] >= M[i, 0] and M[i, 1] >= M[i, 2]:
                    y_hat[i] = 2
                elif M[i, 2] >= M[i, 0] and M[i, 2] >= M[i, 1]:
                    y_hat[i] = 3
                else:
                    y_hat[i] = np.random.choice(3) + 1

            ### Maximization - Step

            # Categorize the data to each class
            c1 = X[y_hat == 1]
            c2 = X[y_hat == 2]
            c3 = X[y_hat == 3]

            # Print out the number of data for each class
            if len(log_likelihoods) == 1:
                print('The number of records in each cluster:')
            print('Cluster 1: %d, Cluster 2: %d, Cluster 3: %d' % (len(c1), len(c2), len(c3)))

            # Calculate the parameters for each class
            mu[0] = [c1[:, 0].mean(), c1[:, 1].mean()]
            mu[1] = [c2[:, 0].mean(), c2[:, 1].mean()]
            mu[2] = [c3[:, 0].mean(), c3[:, 1].mean()]

            sigma[0] = [[c1[:, 0].std(), 0], [0, c1[:, 1].std()]]
            sigma[1] = [[c2[:, 0].std(), 0], [0, c2[:, 1].std()]]
            sigma[2] = [[c3[:, 0].std(), 0], [0, c3[:, 1].std()]]

            w[0] = len(c1) / float(len(X))
            w[1] = len(c1) / float(len(X))
            w[2] = len(c1) / float(len(X))

            # Check for convergence
            if len(log_likelihoods) < 2:
                continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps:
                print('Log likelihood has converged.')
                break

        # Store all results
        self.params = namedtuple('params', ['mu', 'sigma', 'w', 'log_likelihoods', 'num_iters', 'y_hat'])
        self.params.mu = mu
        self.params.sigma = sigma
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)
        self.params.y_hat = y_hat

        return self.params


def iris_2d():
    # Load iris data
    iris = datasets.load_iris()
    X = iris.data[:, [0, 1]]  # Choose first two features for better visualization
    Y = iris.target           # True value
    Y[:] = Y + 1;

    # Fit EM algorithm
    gmm = GMM(3, 0.001)
    params = gmm.EM(X, max_iters=100)

    # Make the dataframe
    dic = {'x1': X.T[0], 'x2': X.T[1], 'label': params.y_hat}
    df = pd.DataFrame(dic)
    groups = df.groupby('label')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for name, group in groups:
        axes[0].plot(group.x1, group.x2, marker='o', linestyle='', label=name)

    axes[0].legend()
    axes[0].set_title('Clustering with EM algorithm')
    axes[0].set_xlabel('sepal_length (cm)')
    axes[0].set_ylabel('sepal_width (cm)')
    # axes[0].set_xlabel('sepal_width (cm)')
    # axes[0].set_xlabel('petal_length (cm)')
    # axes[0].set_ylabel('petal_length (cm)')
    # axes[0].set_ylabel('petal_width (cm)')

    axes[1].plot(np.array(params.log_likelihoods))
    axes[1].set_title('Log Likelihood')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('log likelihood')
    plt.show()


if __name__ == "__main__":
    iris_2d()
