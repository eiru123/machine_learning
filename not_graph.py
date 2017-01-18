import tensorflow as tf

x_data = [[-2., 1.], [-2., -1.], [1., -1.], [1., 1.]]
y_data = [0., 0., 0., 1.]
y = []
a = 0.05
# 논리 연산 스위치
x = tf.Variable(x_data)

# 논리식 결과값
T = tf.Variable(y_data)

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([2]))

# evidence
evidence = tf.reduce_sum(tf.add(tf.matmul(x, W), b))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(4):
    if sess.run(evidence) > 0:
        y.append(1.)
    else:
        y.append(0.)

for step in range(500):
    for j in range(4):
        if sess.run(evidence) > 0:
            y[j] = 1.
        else:
            y[j] = 0.
    sub = tf.sub(T, y)
    # 계산을 위해 shape을 [4] 에서 [4,1]로 확장
    sub_expand = tf.expand_dims(sub, 1)
    train = tf.add(W, tf.matmul(tf.transpose(x), tf.mul(sub_expand, a)))
    update = W.assign(train)
    print(y, sess.run(sub), sess.run(sub_expand))
    print(sess.run(update), sess.run(b), sess.run(W), sess.run(evidence))
