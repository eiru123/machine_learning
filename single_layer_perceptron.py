import tensorflow as tf
import numpy as np

point_a = []
point_b = []

for i in range(500):
    point_a.append([60 * np.random.random_sample(), 60 * np.random.random_sample()])
    point_b.append([60 * np.random.random_sample() + 40, 60 * np.random.random_sample() + 40])

data_set = tf.Variable(point_a + point_b)

x = tf.placeholder("float", [None, 2])
W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))

assignment = []
for i in range(1000):
    if i < 500:
        assignment.append([1, 0])
    else:
        assignment.append([0, 1])

matm = tf.matmul(x, W)
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 2])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
list_data_set = list(sess.run(data_set))
for i in range(1000):
    sess.run(train_step, feed_dict={x: list_data_set, y_: assignment})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    print(sess.run(W), sess.run(b))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: [[130.0, 55.0]], y_: [[0, 1]]}))
