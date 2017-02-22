import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

print(c)

print(sess.run(c))

# placeholder 바로 값을 넣어주는게 아니라 오퍼레이션을 실행할 때 변수들을 넣어 주는 것이다
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 3, b: 4}))

