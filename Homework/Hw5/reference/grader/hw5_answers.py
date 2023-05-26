from cvxopt import matrix
from cvxopt import solvers
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10

class Circles(object):
    def __init__(self, random_state=42):
        self.data, self.labels = make_circles(n_samples=400, noise=0.1, random_state=random_state, factor=0.7)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.labels,
                                                                                test_size=0.33, random_state=42)


class DataBlobs:
    def __init__(self, centers, n_samples=400, n_features=2, random_state=42):
        self.data, self.labels = make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=1.75,
                                            centers=centers, shuffle=False, random_state=random_state)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data, self.labels,
                                                                                test_size=0.33, random_state=42)



class KMeans:
    def __init__(self, k):
        """
    @param k: (int) number of means/centroids to evaluate
    """
        self.k = k
        self.centroids = None
        self.plots = []

    def initialize_centroids(self, points):
        """
    Randomly select k distinct points from the dataset in self.X as centroids
    @param points: (np.ndarray) of dimension (N, d) N is the number of points
    @return: centroids array of shape (k, d)
    """
        centroids = None
        # Workspace 1.1
        # BEGIN
        centroids = points[np.random.choice(range(points.shape[0]), size=self.k, replace=False)]
        # END
        return centroids

    def compute_distances(self, points):
        """
    Compute a distance matrix of size (N, k) where each cell (i, j) represents the distance between
    i-th point and j-th centroid. We shall use Euclidean distance here.
    #WARNING: do not use scikit's BallTree. Only numpy arrays
    @return: distances_matrix : (np.ndarray) of the dimension (N, k)
    """
        distances_matrix = np.zeros((points.shape[0], self.k))
        # Workspace 1.2
        # BEGIN
        gram_matrix = points.dot(self.centroids.T)
        distances_square = (np.linalg.norm(points, axis=1) ** 2)[:, None] + (np.linalg.norm(self.centroids,
                                                                                            axis=1) ** 2)[None, :] \
                           - 2 * gram_matrix
        distances_matrix = np.abs(distances_square) ** 0.5  ## abs because of floating errors (small negative value)
        # END
        return distances_matrix

    @staticmethod
    def compute_assignments(distances_to_centroids):
        """
    Compute the assignment array of shape (N,) where assignment[i] = j if and only if point i
    belongs to the cluster of centroid j
    @param distances_to_centroids: The computed pairwise distances matrix of shape (N,k)
    @return: assignments array of shape (N,)
    """
        assignments = np.zeros((distances_to_centroids.shape[0],))

        # Workspace 1.3
        # BEGIN
        assignments = np.argmin(distances_to_centroids, axis=1)
        # END
        return assignments

    def compute_centroids(self, points, assignments):
        """
    Given the assignments array for the points, compute the new centroids
    @param assignments: array of shape (N,) where assignment[i] is the current cluster of point i
    @return: The new centroids array of shape (k,d)
    """
        # Workspace 1.4
        centroids = np.zeros((self.k, points.shape[1]))
        # BEGIN

        for i in range(self.k):
            elements = (assignments == i)
            if np.sum(elements) != 0:
                centroids[i] = np.mean(points[elements], axis=0)
        # END
        return centroids

    def compute_objective(self, points, assignments):
        return np.sum(np.linalg.norm(points - self.centroids[assignments], axis=1) ** 2)

    def fit(self, points, tol=1e-2):
        """
    Implement the K-means algorithm here as described above. Loop untill the improvement of the objective
    is lower than tol. At the end of each iteration, save the k-means objective and return the objective values
    at the end

    @param points:
    @return:
    """
        self.centroids = self.initialize_centroids(points)
        objective = np.inf
        assignments = np.zeros((points.shape[0],))
        history = []

        # Workspace 1.5

        while True:
            # BEGIN
            distances_to_centroids = self.compute_distances(points)
            assignments = self.compute_assignments(distances_to_centroids)
            self.centroids = self.compute_centroids(points, assignments)
            new_objective = self.compute_objective(points, assignments)
            if np.abs(objective - new_objective) < tol:
                break
            objective = new_objective
            history.append(objective)
            # END
        return history

    def predict(self, points):
        # Workspace 1.6
        assignments = np.zeros((points.shape[0],))
        # BEGIN
        distances_to_centroids = self.compute_distances(points)
        assignments = self.compute_assignments(distances_to_centroids)
        # END
        return assignments


def evaluate_clustering(trained_model, data, labels):
    # We can assume that the number of clusetrs and the number of class labels are the same
    clusters = trained_model.predict(data)
    # Workspace 1.7
    # BEGIN
    # create the confusion matrix and use `linear_sum_assignment` to find X_hat
    # Transform the cluster assignments to predicted labels using X_hat then compute and return the accuracy
    accuracy = 0
    confusion = np.eye(5)[clusters].T.dot(np.eye(5)[labels])
    _, col_ind = linear_sum_assignment(cost_matrix=confusion, maximize=True)
    predicted_labels = col_ind[clusters]
    accuracy = np.mean(predicted_labels == labels)
    ""
    # END
    return accuracy


class KMeansPP(KMeans):

    def initialize_centroids(self, points):
        # Workspace 1.9.a
        # Complete K-means++ centroid initialization. The first step (first centroid) is provided in the next line
        # Hint: You can modify self.centroids and use self.compute_distances to avoid re-coding distances computations
        centroids = points[np.random.choice(range(points.shape[0]), size=1)]
        # BEGIN
        self.centroids = centroids
        for _ in range(self.k - 1):
            distances_squared = self.compute_distances(points) ** 2
            min_distances_squared = np.min(distances_squared, axis=1)
            probas = min_distances_squared / np.sum(min_distances_squared)
            new_centroid = points[np.random.choice(range(points.shape[0]), size=1, p=probas)]
            self.centroids = np.concatenate([self.centroids, new_centroid], axis=0)
        centroids = self.centroids
        # END
        return centroids

class LinearKernel(object):
    def compute(self, x1, x2):
        """
    Compute the kernel matrix
    @param x1: array of shape (m1,p)
    @param x2: array of shape(m2,p)
    @return: K of shape (m1,m2) where K[i,j] = <x1[i], x2[j]>
    """
        # Workspace 2.1.a
        K = np.zeros((x1.shape[0], x2.shape[0]))
        # BEGIN
        K = x1.dot(x2.T)
        # END
        return K


class RadialKernel(object):

    def __init__(self, gamma):
        self.gamma = gamma

    def compute(self, x1, x2):
        """
    Compute the kernel matrix. Hint: computing the squared distances is similar to compute_distances in K-means
    @param x1: array of shape (m1,p)
    @param x2: array of shape(m2,p)
    @return: K of shape (m1,m2) where K[i,j] = K_rad(x1[i],x2[j]) = exp(-gamma * ||x1[i] - x2[j]||^2)
    """
        # Workspace 2.1.b
        K = np.zeros((x1.shape[0], x2.shape[0]))
        # BEGIN
        gram_matrix = x1.dot(x2.T)
        distances_square = (np.linalg.norm(x1, axis=1) ** 2)[:, None] + (np.linalg.norm(x2, axis=1) ** 2)[None, :] \
                           - 2 * gram_matrix
        K = np.exp(-self.gamma * distances_square)
        # END
        return K


class PolynomialKernel(object):

    def __init__(self, c, p):
        self.c = c
        self.p = p

    def compute(self, x1, x2):
        """
    Compute the kernel matrix.
    @param x1: array of shape (m1,p)
    @param x2: array of shape(m2,p)
    @return: K of shape (m1,m2) where K[i,j] = (x1[i].x2[j] + c)^p
    """
        # Workspace 2.1.b
        K = np.zeros((x1.shape[0], x2.shape[0]))
        # BEGIN
        K = (x1.dot(x2.T) + self.c) ** self.p
        # END
        return K






def quadratic_solver(K, y):
    """
Solve for alpha of the dual problem,
@param K: Kernel matrix K of shape (m,m)
@param y: labels array y of shape (m,)
@return: optimal alphas of shape (m,)
"""

    # Workspace 2.2
    m = K.shape[0]
    # P = ? # shape (m,m)
    # q = ? # shape(m,1)
    # G = ? # shape(m,m)
    # h = ? # shape (m,)
    # A = ? # shape (1,m)
    # b = ? # scalar
    # BEGIN
    P = (K * y.reshape(-1, 1)) * y.reshape(1, -1)  # shape (m,m)
    q = -np.ones((m, 1))  # shape(m,1)
    G = -np.eye(m)  # shape(m,m)
    h = 0.0 * np.zeros(m)  # shape (m,)
    A = 1.0 * y.reshape(1, -1)  # shape (1,m)
    b = 0.0  # scalar

    # END
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alphas = np.array(sol['x'])
    alphas = alphas * (np.abs(alphas) > 1e-8)  # zeroing out the small values
    return alphas.reshape(-1)


class SVM(object):

    def __init__(self, kernel):
        self.kernel = kernel
        self.X = None
        self.y = None
        self.intercept = None
        self.alphas = None

    def fit(self, X, y):
        """
    Transform y to (-1,1) and use self.kernel to compute K
    Solve for alphas and compute the intercept using the provided expression
    Keep track of X and y since you'll need them for the prediction
    @param X: data points of shape (m,p)
    @param y: (0,1) labels of shape (m,)
    @return: None
    """
        # Workspace 2.3
        self.X = X
        self.y = 2 * y - 1
        # BEGIN
        K = self.kernel.compute(X, X)
        self.alphas = quadratic_solver(K, self.y)
        self.intercept = 0
        for m in np.where(self.alphas != 0)[0]:
            self.intercept += self.y[m] - np.sum(self.alphas * self.y * K[:, m])
        self.intercept = self.intercept / np.sum(self.alphas != 0)
        # END

    def predict(self, X):
        """
    Predict the labels of points in X
    @param X: data points of shape (m,p)
    @return: predicted 0-1 labels of shape (m,)
    """
        # Workspace 2.4
        predicted_labels = np.zeros((X.shape[0],))
        # BEGIN
        predicted_sign = np.sign(
            np.sum((self.y * self.alphas).reshape(-1, 1) * self.kernel.compute(self.X, X), axis=0)
            + self.intercept)
        predicted_labels = (1 + predicted_sign) / 2
        # END
        return predicted_labels


class MSE(object):
    def __init__(self):
        self.saved_arrays = None

    def forward(self, y_pred, y_true):
        """
    Compute the MSE loss
    @param y_pred: shape (m,q)
    @param y_true: shape (m,q)
    @return: scalar
    """
        # Workspace 3.1
        mse = 0
        # BEGIN
        self.saved_arrays = [y_pred, y_true]
        mse = np.sum(np.mean((y_pred - y_true) ** 2, axis=0))
        # END
        return mse

    def backward(self):
        """
    Compute the gradient w.r.t to the prediction y_pred
    You'll have to cache the necessary quantities into your object-level
        `saved_arrays` variable during the forward pass
    @return: shape (m,q)
    """
        # Workspace 3.2
        grad_input = None
        # BEGIN
        y_pred, y_true = self.saved_arrays
        self.saved_arrays = None
        grad_input = 2 * (y_pred - y_true) / y_pred.shape[0]
        # END
        return grad_input


class Layer(object):
    """
Template Layer that will be used to implement all other layers
"""

    def __init__(self):
        self.saved_arrays = []  # You might need them for the backward pass
        self.parameters = None

    def forward(self, x):
        """
In the forward pass we receive an array containing the input and return an array containing the output.
You can cache arbitrary objects for use in the backward pass in self.saved_arrays
@param x: input array of size (batch_size, d)
@return: output array
"""
        self.saved_arrays = [x]
        return x

    def backward(self, grad_output):
        """
In the backward pass we receive an array containing the gradient of the loss with respect to the output,
and we need to compute the gradient of the loss with respect to the input and the gradient of the weights (default as 0)
@param grad_output:
@return: grad
"""
        saved, = self.saved_arrays
        grad_input = 0
        grad_parameters = 0

        return grad_output, grad_parameters

    def apply_gradient(self, gradients):
        """
Method to apply the gradients to the layer parameters. It will be called by the optimizer.
@param gradients:
@return: None
"""
        pass


class Sigmoid(Layer):

    def forward(self, x):
        """
Apply the sigmoid function to x. Don't forget to clip x to the interval [-25.0, 25.] before applying the activation
@param x: input array of shape (m,k)
@return: element-size sigmoid of shape (m,k)
"""
        sigmoid = np.zeros_like(x)
        # Workspace 3.3.a
        # BEGIN
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))
        self.saved_arrays = [sigmoid]
        # END
        return sigmoid

    def backward(self, grad_output):
        """
Compute the grad_input and grad_parameters. Activatins don't have parameters, so 0 gradient is the default
@param grad_output: input array of shape (m,k)
@return: tuple (grad_input of shape element-size sigmoid of shape (m,k), 0)
"""
        grad_parameters = 0
        grad_input = 0

        # Workspace 3.3.b
        # BEGIN
        sigmoid, = self.saved_arrays
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        # END
        return grad_input, grad_parameters


class ReLU(Layer):

    def forward(self, x):
        """
Apply the ReLU function to x. Don't forget to clip x to the interval [-25.0, 25.] before applying the activation
@param x: input array of shape (m,k)
@return: element-size ReLU of shape (m,k)
"""
        relu = 0
        # Workspace 3.4.a
        # BEGIN
        relu = x * (x > 0)
        self.saved_arrays = [x]
        # END
        return relu

    def backward(self, grad_output):
        """
Compute the grad_input and grad_parameters. Activatins don't have parameters, so 0 gradient is the default
@param grad_output: input array of shape (m,k)
@return: tuple (grad_input of shape element-size sigmoid of shape (m,k), 0)
"""
        grad_input = 0
        grad_parameters = 0
        # Workspace 3.4.b
        # BEGIN
        x, = self.saved_arrays
        grad_input = grad_output * (x > 0)
        # END
        return grad_input, grad_parameters


class Dense(Layer):

    def __init__(self, input_dimension, output_dimension):
        """
Initialize the parameters
@param input_dimension: The dimension of the input data
@param output_dimension: the dimension of the output
"""
        super(Dense, self).__init__()
        # Workspace 3.5.a
        # Intialize the bias and weights using random normal distibution
        # You should scale each array by 1 / sqrt(N) where N is the number of elements in the array
        self.bias = None
        self.weights = None
        # BEGIN
        self.bias = np.random.randn(1, output_dimension) / output_dimension ** 0.5
        self.weights = np.random.randn(input_dimension, output_dimension) / (
                    output_dimension * input_dimension) ** 0.5
        # END

    def forward(self, x):
        """
Apply the linear projection
@param x of shape (m,input_dimension)
@param z = xw + b of shape (m,output_dimension)
"""
        output = 0
        # Workspace 3.5.b
        # BEGIN
        self.saved_arrays = [x]
        output = x.dot(self.weights) + self.bias
        # END
        return output

    def backward(self, grad_output):
        """
Compute the gradients using the aforementioned formulas. Do not change the return signature
@param grad_output: shape (m, output_dimension)
@return: a tuple (grad_input, grad_weights, grad_bias)
        grad_input of shape (m, input_dimension)
        grad_weights of shape (input_dimension, output_dimension)
        grad_bias of shape (1, output_dimension)
"""
        grad_input = 0
        grad_weights = 0
        grad_bias = 0
        # Workspace 3.5.c
        # BEGIN
        x, = self.saved_arrays
        grad_input = grad_output.dot(self.weights.T)
        grad_weights = x.T.dot(grad_output)
        grad_bias = grad_output.sum(0).reshape(1, -1)
        # END
        return grad_input, grad_weights, grad_bias

    def apply_gradient(self, gradients):
        grad_weights, grad_bias = gradients
        self.weights -= grad_weights
        self.bias -= grad_bias


class SGD(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.layers = None
        self.gradients = None

    def set_layers(self, layers):
        """
    Saves the layers stack
    @param layers: list of Layer instances (the same stack stored in the network)
    @return: None
    """
        self.layers = layers

    def set_gradients(self, gradients):
        """
    Saves the layers' parameters gradients before applying them.
    self.layers and self.gradients have the same size. self.gradients is a list of lists
    self.gradients[i] contains the list of parameters from self.layers.backward
    @param gradients: List of parameters gradients
    @return:
    """
        self.gradients = gradients

    def apply_gradients(self):
        """
    Multiply the gradients by the learning_rate before passing them to apply_gradient of the layers
    We loop through self.gradients (List of layers gradients computed from the backward pass).
    Then we call the corresponding layers apply_gradient with the scaled gradients.
    Hint: gradients[i] is a list on numpy as arrays that correspond to grad_parameters of layers[i]
    @return: None
    """
        # Workspace 3.6
        # BEGIN
        for layer, layer_gradients in zip(self.layers, self.gradients):
            layer.apply_gradient([self.learning_rate * g for g in layer_gradients])
        # END
        self.gradients = None


class Network(object):
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = []
        self.optimizer.set_layers(self.layers)

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        return self.forward(x)

    def forward(self, x):
        output = x
        # Workspace 3.7
        # BEGIN
        for layer in self.layers:
            output = layer.forward(output)
        # END
        return output

    def backward(self):

        grad_output = self.loss.backward()
        parameters_gradients = []

        for layer in self.layers[::-1]:
            grads = layer.backward(grad_output)
            grad_output = grads[0]
            parameters_gradients.append(grads[1:])
        parameters_gradients = parameters_gradients[::-1]
        return parameters_gradients

    def fit(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        parameters_gradients = self.backward()
        self.optimizer.set_gradients(parameters_gradients)
        self.optimizer.apply_gradients()
        return loss



class Adam(SGD):
    def __init__(self, learning_rate, beta_1, beta_2):
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_moment = None
        self.second_moment = None
        self.time_step = 0

    def apply_gradients(self):
        # Workspace 3.10.a
        # BEGIN

        self.time_step += 1
        if self.first_moment is None:
            self.first_moment = [[g * 0 for g in layer_gradients] for layer_gradients in self.gradients]
            self.second_moment = [[g * 0 for g in layer_gradients] for layer_gradients in self.gradients]

        for layer, layer_gradients, moment_1, moment_2 in zip(self.layers, self.gradients, self.first_moment,
                                                              self.second_moment):
            adam_gradients = []
            for i in range(len(layer_gradients)):
                moment_1[i] = self.beta_1 * moment_1[i] + (1 - self.beta_1) * layer_gradients[i]
                moment_2[i] = self.beta_2 * moment_2[i] + (1 - self.beta_2) * layer_gradients[i] ** 2
                moment_1_estimate = moment_1[i] / (1 - self.beta_1 ** self.time_step)
                moment_2_estimate = moment_2[i] / (1 - self.beta_2 ** self.time_step)
                adam_gradients.append(
                    self.learning_rate * moment_1_estimate / (moment_2_estimate ** 0.5 + 1e-8))
            layer.apply_gradient(adam_gradients)
        # END
        self.gradients = None


circles = Circles()
multi_blobs = DataBlobs(centers=5)
binary_blobs = DataBlobs(centers=2)
