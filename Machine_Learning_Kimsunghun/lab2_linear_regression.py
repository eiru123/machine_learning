import tensorflow as tf

# dataset
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W, b random value setting
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# data 재사용을 위해  placeholder 사용
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# our Hypothesis (y = Wx + b)
hypothesis = W * X + b

# cost function - calculate difference real value and hypothesis value
cost = tf.reduce_mean(tf.square(tf.sub(hypothesis, Y)))

# cost minimize
a = tf.Variable(0.1)  # learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# initializer
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

# placeholder 를 이용하면 Hypothesis 그대로 사용해서 예측값을 구하는데 사용할 수 있다
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))