import tensorflow as tf
import numpy as np

# unpack=True 이면 받아온 데이터 열을 transpose 해준다
xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data_load = xy[0:-1]
y_data_load = xy[-1]
print(xy, x_data_load, y_data_load)
# data set
x_data = [[1., 1., 1., 1., 1.],  # b 를 합치기 위해서 모두 1로 넣어준다.
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]

y_data = [1, 2, 3, 4, 5]

# weight, b setting
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# hypothesis
hypothesis = tf.matmul(W, x_data)  # tf.matmul() 함수는 int 타입으로는 계산을 못하므로 float 으로 바꿔준다.

# cost
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize
a = tf.Variable(0.1)  # learning rate 
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
