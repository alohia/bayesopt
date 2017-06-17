

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.gaussian_process import GaussianProcessRegressor

#MNIST dataset:
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Train set: 55000, Test: 10000
##Example of an image plot:
label0  = tf.arg_max(mnist.train.labels[0],0).eval()
im0 = mnist.train.images[0]
plt.imshow(im0.reshape([28,28]), cmap = 'gray')
plt.show()


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

def nn_train(h_dim):
    y = neural_network(x, h_dim)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.MomentumOptimizer(0.5, 0.5).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for _ in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100) #These variables are numpy arrays
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


sess = tf.InteractiveSession()
nn_train(10)

##Gaussian Process
