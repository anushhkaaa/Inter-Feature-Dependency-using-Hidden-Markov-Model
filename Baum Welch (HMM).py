import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\anush\\Downloads\\train_filter_2.csv')
df = pd.DataFrame(data, columns = data.columns)
#df = df.drop(columns=['Unnamed: 0', 'FileName','timestamp', 'City', 'weather', 'Fron_vehicle','stopsign','motorcycle','bicycle'])
df = df.drop(columns=['FileName','timestamp', 'City', 'weather', 'Fron_vehicle','stopsign','motorcycle','bicycle'])
df.isnull().values.any()
df = df.dropna()
V = df.to_numpy().astype(int)
#print(V)
print(V.shape)
V[V > 3] = 1  #thresholding 
states = V.shape[1]
out_states = 4

# HMM initialisation

# initialise initial probabilities
r = np.random.rand(states)
startprobs = r/sum(r)
startprobs = np.around(startprobs, decimals=3)

# initialise transition probabilities
transmat = np.zeros((states, states))
for i in range(states):
    r =np.random.rand(states)
    transmat[i] = r/sum(r)
transmat = np.around(transmat, decimals=3)
#print(trans)

#initialise emission probabilities
emis = np.zeros((states, out_states))
for i in range(states):
    r = np.random.rand(out_states)
    emis[i] = r/sum(r)

def forward(V, transmat, emis, startprob):
    alpha = np.zeros((V.shape[0], transmat.shape[0]))
    #alpha = np.zeros(V.shape)
    alpha[0, :] = startprob * emis[:, 0]
 
    for t in range(1, V.shape[0]):
        for j in range(transmat.shape[0]):
            for k in range(4):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
             alpha[t, j] = alpha[t - 1].dot(transmat[:, j]) * emis[j, k]
            #alpha[t, j] = alpha[t - 1].dot(transmat[:, j])
 
     
    return alpha

def backward(V, transmat, emis):
    beta = np.zeros((V.shape[0], transmat.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((transmat.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(transmat.shape[0]):
            for k in range(4):
                beta[t, j] = (beta[t + 1] * emis[:,k]).dot(transmat[j, :])
            #beta[t, j] = beta[t + 1].dot(transmat[j, :])
 
    return beta

def baum_welch(V, transmat, emis, startprobs, n_iter=100):
    M = transmat.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, transmat, emis, startprobs)
        beta = backward(V, transmat, emis)
 
        xi = np.zeros((M, M, T - 1))  #shape: (21,21,197)
        denominator = np.float(0.0)
        numerator = np.float(0.0)
        for t in range(T - 1):
            for k in range(4):
                denominator = np.dot(np.dot(alpha[t, :].T, transmat) * emis[:, k].T, beta[t + 1, :])
                # print(denominator, type(denominator))
            
            #denominator = np.dot(np.dot(alpha[t, :].T, transmat) ,beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * transmat[i, :] * emis[:, k] * beta[t + 1, :].T
                #numerator = alpha[t, i] * transmat[i, :]*beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        #print(denominator.shape)
        gamma = np.sum(xi, axis=1)
        #print(gamma.shape)
        transmat = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        #print(gamma.shape)
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        # print(gamma.shape)
        #print(gamma.shape)
        K = emis.shape[1]
        #print(K, emis.shape)
        
        denominator = np.sum(gamma, axis=1)
        #print(denominator.shape)
        for l in range(K):
            emis[:, l] =  np.sum(gamma[:, l], axis=0)
 
        covar = np.divide(emis, denominator.reshape(-1,1)) ## denominator 21*1 emis 21*4  gamma 21*198
        #print(emis.shape, gamma.shape)
    dict1 = dict()
    dict1["transmat"] = transmat
    dict1["covar"] = covar
    return(dict1) 

output = baum_welch(V,transmat,emis,startprobs,100)
output["transmat"].shape
output["covar"].shape
print(baum_welch(V,transmat,emis,startprobs,100))