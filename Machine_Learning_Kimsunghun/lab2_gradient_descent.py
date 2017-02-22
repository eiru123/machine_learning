import tensorflow as tf
import matplotlib.pyplot as plt


# data set
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W, b random value setting
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

# data 재사용을 위해  placeholder 사용
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# our Hypothesis (y = Wx + b)
hypothesis = tf.mul(W, X)

# cost function - calculate difference real value and hypothesis value
cost = tf.reduce_mean(tf.square(tf.sub(hypothesis, Y)))

# cost minimize
decsent = tf.sub(W, tf.mul(0.1, tf.reduce_mean(tf.mul(tf.sub(hypothesis, Y), X))))
update = W.assign(decsent)

# initializer
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

# placeholder 를 이용하면 Hypothesis 그대로 사용해서 예측값을 구하는데 사용할 수 있다
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(hypothesis, feed_dict={X: x_data}))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()