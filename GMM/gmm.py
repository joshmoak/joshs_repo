# solutions.py
"""Volume 3: Gaussian Mixture Models. Solutions File."""

import numpy as np
from scipy import stats as st
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture


class GMM:
    # Problem 1
    def __init__(self, n_components, weights=None, means=None, covars=None):
        """
        Initializes a GMM.
        
        The parameters weights, means, and covars are optional. If fit() is called,
        they will be automatically initialized from the data.
        
        If specified, the parameters should have the following shapes, where d is
        the dimension of the GMM:
            weights: (n_components,)
            means: (n_components, d)
            covars: (n_components, d, d)
        """

        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.covars = covars

    # Problem 2
    def component_logpdf(self, k, z):
        """
        Returns the logarithm of the component pdf. This is used in several computations
        in other functions.
        
        Parameters:
            k (int) - the index of the component
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the log pdf of the component at 
        """

        mu = self.means[k]
        cov = self.covars[k]
        w = self.weights

        draw = st.multivariate_normal.logpdf(z, mu, cov)

        return np.log(w[k]) + draw
        
    # Problem 2
    def pdf(self, z):
        """
        Returns the probability density of the GMM at the given point or points.
        
        Parameters:
            z ((d,) or (..., d) ndarray) - the point or points at which to compute the pdf
        Returns:
            (float or ndarray) - the value of the GMM pdf at z
        """

        draws = np.array([st.multivariate_normal.pdf(z, self.means[i], self.covars[i]) for i in range(len(self.means))])

        return self.weights @ draws

    # Problem 3
    def draw(self, n):
        """
        Draws n points from the GMM.
        
        Parameters:
            n (int) - the number of points to draw
        Returns:
            ((n,d) ndarray) - the drawn points, where d is the dimension of the GMM.
        """
        
        X = np.random.choice(self.n_components, p = self.weights, size = n)

        draws = np.array([st.multivariate_normal.rvs(self.means[x], self.covars[x]) for x in X])
        return draws
        
    # Problem 4
    def _compute_e_step(self, Z):
        """
        Computes the values of q_i^t(k) for the given data and current parameters.
        
        Parameters:
            Z ((n, d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components, n) ndarray): an array of the computed q_i^t(k) values, such
                    that result[k,i] = q_i^t(k).
        """

        l = np.array([self.component_logpdf(k,Z) for k in range(self.n_components)]).T
        L = np.max(l, axis = 1)
        numer = np.exp(l.T - L)
        Q = numer / np.sum(numer.T,  axis = 1)
        return Q

    # Problem 5
    def _compute_m_step(self, Z):
        """
        Takes a step of the expectation maximization (EM) algorithm. Return
        the updated parameters.
        
        Parameters:
            Z (n,d) ndarray): the data that is being used for training; d is the
                    dimension of the data.
        Returns:
            ((n_components,) ndarray): the updated component weights
            ((n_components,d) ndarray): the updated component means
            ((n_components,d,d) ndarray): the updated component covariance matrices
        """

        n,d = Z.shape

        Q = self._compute_e_step(Z)
        # Update weights
        w = Q.sum(axis = 1) / n

        # Update means
        mu = np.dot(Q,Z) / Q.sum(axis = 1).reshape(-1,1)

        # update covars
        new_Z = np.expand_dims(Z, 0)
        new_mu = np.expand_dims(mu, 1)
        
        numer = np.einsum("Kn, Knd, KnD -> KdD", Q, new_Z - new_mu, new_Z - new_mu)
        covar = numer / np.sum(Q, axis = 1).reshape((-1,1,1))

        return w, mu, covar
        
    # Problem 6
    def fit(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model by applying the Expectation Maximization algorithm until the
        parameters appear to converge.
        
        Parameters:
            Z ((n,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            self
        """
        n, d = Z.shape
        if self.weights is None:
            self.weights = np.ones(self.n_components) / self.n_components

        if self.means is None:
            k = np.random.randint(0,n, self.n_components)
            self.means = Z[k]

        if self.covars is None:
            var = np.var(Z, axis = 0)
            d = np.diag(var)
            self.covars = np.array([d for _ in range(self.n_components)])

        for i in range(maxiter):
            print(f"iteration : {i}", end = "\r")

            new_weights, new_means, new_covars = self._compute_m_step(Z)

            change = (np.max(np.abs(new_weights - self.weights))
                    + np.max(np.abs(new_means - self.means))
                    + np.max(np.abs(new_covars - self.covars)))

            if change < tol:
                return self

            self.weights, self.means, self.covars = new_weights, new_means, new_covars
            
   
        return self
        
    # Problem 8
    def predict(self, Z):
        """
        Predicts the labels of data points using the trained component parameters.
        
        Parameters:
            Z ((m,d) ndarray): the data to label; d is the dimension of the data.
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        labels = []
        for z in Z:
            which_dist = []
            for k in range(self.n_components):
                logpdf = self.component_logpdf(k,z)
                which_dist.append(logpdf)
            labels.append(np.argmax(np.array([which_dist])))
        return labels

    def fit_predict(self, Z, tol=1e-3, maxiter=200):
        """
        Fits the model and predicts cluster labels.
        
        Parameters:
            Z ((m,d) ndarray): the data to use for training; d is the
                dimension of the data.
            tol (float): the tolderance to check for convergence
            maxiter (int): the maximum number of iterations allowed
        Returns:
            ((m,) ndarray): the predicted labels of the data
        """
        return self.fit(Z, tol, maxiter).predict(Z)

# Problem 3
def problem3():
    """
    Draw a sample of 10,000 points from the GMM defined in the lab pdf. Plot a heatmap
    of the pdf of the GMM (using plt.pcolormesh) and a hexbin plot of the drawn points.
    How do the plots compare?
    """
    
    gmm = GMM(n_components = 2, 
            weights = np.array([0.6, 0.4]),
            means = np.array([[-0.5, -4.0], [0.5, 0.5]]),
            covars = np.array([
                [[1,0],[0,1]],
                [[0.25, -1], [-1, 8]]
            ]))

    samples = gmm.draw(10000)
    x = np.linspace(-8,8,100)
    y = np.linspace(-8,8,100)
    X, Y = np.meshgrid(x, y)


    Z = np.array([[
        gmm.pdf([X[i,j], Y[i,j]]) for j in range(100)
        ] for i in range(100)
        ])
    plt.figure(figsize = (15,5))
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, shading = "auto")
    plt.title("PDF")
    plt.subplot(122)
    plt.hexbin(samples[:,0], samples[:,1])
    plt.title("Draws from GMM")
    plt.show()
    
# Problem 7
def problem7(filename='problem7.npy'):
    """
    The file problem6.npy contains a collection of data drawn from a GMM.
    Train a GMM on this data with n_components=3. Plot the pdf of your
    trained GMM, as well as a hexbin plot of the data.
    """
    
    data = np.load(filename)
    
    gmm = GMM(3)
    gmm.fit(data)

    samples = gmm.draw(10000)
    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)
    X, Y = np.meshgrid(x, y)


    Z = np.array([[
        gmm.pdf([X[i,j], Y[i,j]]) for j in range(100)
        ] for i in range(100)
        ])
    plt.figure(figsize = (15,5))
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, shading = "auto")
    plt.title("PDF")
    plt.subplot(122)
    plt.hexbin(samples[:,0], samples[:,1])
    plt.title("Draws from GMM")
    plt.show()
    
# Problem 8
def get_accuracy(pred_y, true_y):
    """
    Helper function to calculate the actually clustering accuracy, accounting for
    the possibility that labels are permuted.
    
    This computes the confusion matrix and uses scipy's implementation of the Hungarian
    Algorithm (linear_sum_assignment) to find the best combination, which is generally
    much faster than directly checking the permutations.
    """
    # Compute confusion matrix
    cm = confusion_matrix(pred_y, true_y)
    # Find the arrangement that maximizes the score
    r_ind, c_ind = linear_sum_assignment(cm, maximize=True)
    return np.sum(cm[r_ind, c_ind]) / np.sum(cm)
    
def problem8(filename='classification.npz'):
    """
    The file classification.npz contains a set of 3-dimensional data points "X" and 
    their labels "y". Use your class with n_components=4 to cluster the data.
    Plot the points with the predicted and actual labels, and compute and return
    your model's accuracy. Be sure to check for permuted labels.
    
    Returns:
        (float) - the GMM's accuracy on the dataset
    """
    data = np.load(filename)

    X = data["X"]
    y = data["y"]
    gmm = GMM(4)
    labels = gmm.fit_predict(X)
    accuracy = get_accuracy(labels, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], c = labels)

    ax2 = fig.add_subplot(1,2,2,projection = "3d")
    ax2.scatter(X[:,0], X[:,1], X[:,2], c = y)

    plt.show()
    return accuracy

# Problem 9
def problem9(filename='classification.npz'):
    """
    Again using classification.npz, compare your class, sklearn's GMM implementation, 
    and sklearn's K-means implementation for speed of training and for accuracy of 
    the resulting clusters. Print your results. Be sure to check for permuted labels.
    """
    data = np.load(filename)

    X = data["X"]
    y = data["y"]

    start1 = time.time()
    k_labels = KMeans(n_clusters=4).fit_predict(X)
    k_acc = get_accuracy(k_labels, y)
    end1 = time.time() - start1

    start2 = time.time()
    skgmm = GaussianMixture(n_components=4)
    skgmm_labels = skgmm.fit_predict(X)
    skgmm_acc = get_accuracy(skgmm_labels, y)
    end2 = time.time() - start2

    start3 = time.time()
    gmm = GMM(4)
    my_labels = gmm.fit_predict(X)
    my_accuracy = get_accuracy(my_labels, y)
    end3 = time.time() - start3

    print(f"My GMM Method: \n\t Time: {np.round(end3, 3)}\n\t Accuracy: {my_accuracy}\n")
    print(f"Sklearn Kmeans Method: \n\t Time: {np.round(end1, 3)}\n\t Accuracy: {k_acc}\n")
    print(f"SKlearn GMM Method: \n\t Time: {np.round(end2, 3)}\n\t Accuracy: {skgmm_acc}\n")




