import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import  MinMaxScaler


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
    

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_item_user(user_arr, item):
    user_item = []
    for user in user_arr:
        cos_sim = dot(user, item.T)/(norm(user)*norm(item))
        user_item.append(cos_sim)

    return user_item

def scale_item_user(item_user):
    scaler_MinMax = MinMaxScaler()
    for i in range(0, len(item_user)):
        reshaped_item = item_user[i].reshape(-1,1)
        scaled_item = scaler_MinMax.fit_transform(reshaped_item).reshape(1,-1)[0]
        item_user[i] = scaled_item

    return item_user

def get_collab_content(item_user_arr, user_user_arr):
    collab_content = np.array(item_user_arr) + np.array(user_user_arr)
    return collab_content

def get_recommendation(user_index, item_item_matrix, collab_content):
    user_arr = collab_content[user_index]
    recommendation = user_arr.index(max(user_arr))
    print(recommendation)
    n_recommendation = sorted(range(len(item_item_matrix[recommendation])), key=lambda i: item_item_matrix[recommendation][i], reverse = True)[:5]
    #print(item_item_matrix[recommendation])
    #print(sorted(item_item_matrix[recommendation], reverse=True))
    print(n_recommendation)
    return n_recommendation
