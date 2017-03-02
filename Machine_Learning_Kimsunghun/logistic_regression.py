import tensorflow as tf
import numpy as np

# unpack=True 이면 받아온 데이터 열을 transpose 해준다 data set
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]


# weight, b setting
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# hypothesis
h = tf.matmul(W, X)  # tf.matmul() 함수는 int 타입으로는 계산을 못하므로 float 으로 바꿔준다.
hypothesis = tf.div(1., 1. + tf.exp(-h))  # sigmoid function

# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
a = tf.Variable(0.1)  # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print('---------------------------------------------')
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5)

print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5)
