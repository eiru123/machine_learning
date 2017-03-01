import tensorflow as tf
import matplotlib.pyplot as plt

# data set
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_sample = len(X)

# set model weight
W = tf.placeholder(tf.float32)

# construct linear model
hypothesis = tf.mul(X, W)

# cost function
cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / m

# initializer
init = tf.global_variables_initializer()

# For_graph
W_val = []
cost_val = []

# launch the graph
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print(i * 0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

# graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost_val')
plt.xlabel('W_val')
plt.show()