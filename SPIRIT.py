import numpy as np
import math
import csv


def calculate_covariance_matrix(X, Y=None):
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

def PCA_transform(X, n_components):
        """ Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset """
        covariance_matrix = calculate_covariance_matrix(X)
        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # eigenvalues 1xn array, eigenvectors nxn matrix
        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]
        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)
        return X_transformed

def TrackW(x_t, W, Y, K, lamda, n):
    '''step 0 and 1:
         Initialize hidden variables W
         Initialize D to small positive value (called spv)
         X_r refers to x´ in paper
         E refers to e in the paper
         Y refers to y in the paper
    '''
    D = []
    spv = 0.05
    for i in range(K):
        D.append(spv)

    X_r = np.zeros((K+1,len(x_t)))
    for i in range(len(x_t)):
        X_r[0][i] = x_t[i]

    E = np.zeros((K,n))

    #step 2: perform updates
    for i in range(K):
        y = 0
        for j in range(n):
            y += W[i][j] * X_r[i][j] #step 2.1
            Y.append(y)
        Y_arr = np.array(Y)
        D[i] = lamda * D[i] + Y_arr[i]*Y_arr[i] #step 2.2
        for j in range(n):
            E[i][j] = X_r[i][j] - Y_arr[i]*W[i][j] #step 2.3
        for j in range(n):
            W[i][j] += (Y_arr[i]*E[i][j])/D[i] #step 2.4
        for j in range(n):
            X_r[i+1][j] = X_r[i][j] - Y_arr[i]*W[i][j] #step 2.5


def SPIRIT(X, T, n):
    #step 0: Initialize
    K = 1
    E = 0
    E_r = []
    for i in range(K):
        E_r.append(0)
    W = np.eye(K, n, dtype=float)

    for t in range(T):
        Y = []
        #step 1: As each new point arrives, update W
        TrackW(X[:,t], W, Y, K, lamda=0.96, n = n)
        #step 2: Update the estimates
        x_t_abs = np.linalg.norm(X[:,t])
        E = (t*E + math.pow(x_t_abs,2)) / (t+1)
        E_K = 0
        for k in range(K):
            E_r[k] = (t*E_r[k] + math.pow(Y[k] ,2)) / (t+1)
            E_K += E_r[k]
        #step 3: adjust number of hidden variable K
        if E_K < f_E*E:
            w_kplus1 = []
            for nn in range(n):
                if nn == K:
                    w_kplus1.append(1)
                else:
                    w_kplus1.append(0)
            W = np.insert(W, K, values=w_kplus1,axis=0)
            E_r.append(0)
            K = K + 1
        elif E_K > F_E*E:
            W = np.delete(W, K-1, axis=0)
            E_r[k] = 0
            K = K - 1
    return K, W

def get_data(sensor_id):
    Timestamp = []
    Speed = []
    f = open('data//traffic_data_01_29_2020.csv')
    r_lines = csv.reader(f, delimiter=',')
    for row in r_lines:
        if row[0] == sensor_id:
            Timestamp.append(row[3])
            Speed.append(row[6])
    Speed = list(map(eval, Speed))
    return Timestamp, Speed

# we have a low-energy and a high-energy threshold, f_E and F_E
f_E = 0.95
F_E = 0.98

n = 5  # number of sensors
T = 35  # time tick
sensor_list = ["773505", "716272", "717620", "764794", "776480"]
S = np.empty(shape=(0,T)) # S(n*T) is input sensor speed sequence ordered by time
for i in sensor_list:
    _, S_i = get_data(i)
    S = np.row_stack((S,S_i))


K, W = SPIRIT(S,T,n)
print(K)
print(W)

