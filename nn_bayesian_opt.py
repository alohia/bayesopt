

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import math as mat
import scipy
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

##Gaussian Process:
def gaussian_kernel(x1,x2,noise,length): #Generate the kernel (cov) of the Gaussian Process
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    kernel = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            kernel[i,j] = noise**2*mat.exp(-0.5*((x1[i]-x2[j])/length)**2)
    return kernel

def LLH_GP(x,y,m,noise,length, sf = 0): #Compute the likelihood of the data (add sf if consider noise)
    ker = gaussian_kernel(x,x, noise, length)
    ker = ker+np.diag([sf]*len(x))
    return 1/2*(mat.log(np.linalg.det(ker))+np.dot(np.dot(np.transpose(y-m),
                                                       np.linalg.inv(ker)),(y-m)))

def opt_hyparams(x,y): #Find the hyperparameters that optimize LLH without noise
    ini = np.array([0,1,1])
    opt = optimize.minimize(lambda params: LLH_GP(x, y, params[0], params[1], params[2]),ini)
    params = opt.x
    m = params[0]
    noise = abs(params[1])
    length = abs(params[2])
    sf = 0
    return m, noise, length, sf

def opt_hyparams_noise(x,y): #Find the hyperparameters that optimize LLH with noise
    ini = np.array([0,1,1,1])
    opt = optimize.minimize(lambda params: LLH_GP(x, y, params[0],
                                                  params[1], params[2], params[3]),ini)
    params = opt.x
    m = params[0]
    noise = abs(params[1])
    length = abs(params[2])
    sf = abs(params[3])
    return m, noise, length, sf

def gp_posterior(x, y, xn, m, noise, length, sf = 0):
    kxx = gaussian_kernel(x, x, noise = noise, length = length)
    kxxn = gaussian_kernel(x, xn, noise = noise, length = length)
    kxnx = gaussian_kernel(xn, x, noise = noise, length = length)
    kxnxn = gaussian_kernel(xn, xn, noise = noise, length = length)
    core = np.linalg.inv(kxx + np.diag([sf]*len(x)))
    En = np.dot(np.dot(kxnx, core), y)
    covn = kxnxn - np.dot(np.dot(kxnx, core), kxxn)

    return En, covn

def data_posterior(x, E, cov):
    data = pd.DataFrame({'x': x})
    data['Mean'] = E
    data['StdDev'] = np.diag(cov)
    #Generate the 5 samples as multivariate normals with 0 mean and covariance sigma
    for i in range(5):
        data['y'+str(i)] = np.random.multivariate_normal(E, cov)
    return data

#plot example
#def f1(x):
#    return np.log(x)

#plt.plot(data['x'],data['Mean'], color = 'black', label = 'Mean')
#plt.plot(x,y, 'ro', label = 'Obs')
#plt.plot(data['x'], data['y0'], label = 'y0')
#plt.plot(data['x'], data['y1'], label = 'y1')
#plt.plot(data['x'], data['y2'], label = 'y2')
#plt.plot(data['x'], data['y3'], label = 'y3')
#plt.plot(data['x'], data['y4'], label = 'y4')
#plt.fill_between(data['x'], data['Mean']-data['StdDev'], data['Mean']+data['StdDev'],
#                 color = 'lightgrey')
#plt.legend()
#plt.show()

#MNIST dataset:
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#placeholder:
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

##Variables:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def neural_network(input, h_dim):
    W0 = weight_variable([784, h_dim])
    b0 = bias_variable([h_dim])
    h = tf.nn.relu(tf.matmul(input, W0) + b0)

    W = weight_variable([h_dim, 10])
    b = bias_variable([10])

    y = tf.nn.softmax(tf.matmul(h, W) + b)
    return y

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# In this case tf.reduce_sum sum over the columns (2nd dimension)

def nn_train(learning_rate, h_dim):
    y = neural_network(x, h_dim)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #train_step = tf.train.MomentumOptimizer(0.5, 0.5).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100) #These variables are numpy arrays
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Adquisition Function

def acquisition_fun(x, y, xn, mean_vector, sigma_vector):
    x_best = x[np.argmax(y)]
    y_best = np.max(y)
    gamma = (mean_vector - y_best)/sigma_vector
    af = (mean_vector - y_best)* norm.cdf(gamma) - sigma_vector * norm.pdf(gamma)
    x_next = xn[np.argmax(af)]
    return x_next

sess = tf.InteractiveSession()
h_dim = 30
nn_train(0.1, h_dim)
n = 500
xn = np.linspace(0,10,n)
##Gaussian Process
l_rates = np.array([0.1,7])
f = [nn_train(l, h_dim) for l in l_rates]
n_iter = 5
for i in range(n_iter):
    m, noise, length, sf = opt_hyparams(l_rates,f)
    E, cov = gp_posterior(l_rates, f, xn, m, noise, length, sf)
    data = data_posterior(xn, E, cov)
    next_candidate = acquisition_fun(l_rates, f, xn, np.array(data['Mean']), np.array(data['StdDev']))
    l_rates = np.append(l_rates,next_candidate)
    f = np.append(f,nn_train(next_candidate, h_dim))


    plt.plot(data['x'],data['Mean'], color = 'black', label = 'Mean')
    plt.plot(l_rates,f, 'ro', label = 'Obs')
    plt.fill_between(data['x'], data['Mean']-data['StdDev'], data['Mean']+data['StdDev'],
                 color = 'lightgrey')

plt.legend()
plt.show()
