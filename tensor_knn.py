import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import operator
point_a = []
point_b = []

for i in range(500):
    point_a.append([int(60 * np.random.random_sample()*1000), int(60 * np.random.random_sample()*1000)])
    point_b.append([int((60 * np.random.random_sample() + 40)*1000), int((60 * np.random.random_sample() + 40)*1000)])

data_set = tf.Variable(tf.slice(tf.random_shuffle(point_a + point_b), [0, 0], [len(point_a + point_b), -1]))

in_x = tf.constant([10000, 70000])

distances = tf.reduce_sum(tf.square(tf.sub(data_set, in_x)), 1)

k = 3
print("point_a")
init = tf.global_variables_initializer()

class_count = {}
with tf.Session() as sess:
    sess.run(init)

    list_data = list(sess.run(data_set))
    distances_list = list(sess.run(distances))
    sorted_distances_list = sorted(distances_list)
    print(sorted_distances_list)
    print(sess.run(data_set))
    print(distances_list)

    for i in range(k):
        idx = distances_list.index(sorted_distances_list[i])
        print(idx, sess.run(data_set[idx]), distances_list[idx])
        if list(list_data[idx]) in point_a:
            class_count['a'] = class_count.get('a', 0) + 1
        else:
            class_count['b'] = class_count.get('b', 0) + 1

print(class_count['a'], class_count['b'])
sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

with tf.Session() as sess:
    print(sess.run(in_x), "= %s" % sorted_class_count[0][0])

assignment = []
for data in list_data:
    data = list(data)
    if data in point_a:
        assignment.append(0)
    else:
        assignment.append(1)

list_data = np.array(list_data)
if sorted_class_count[0][0] == 'a':
    in_x_label = 0
else:
    in_x_label = 1
print(in_x_label)
sess = tf.Session()
plt.scatter(list_data[:, 0], list_data[:, 1], c=assignment, s=50, alpha=0.5)
plt.scatter(sess.run(in_x[0]), sess.run(in_x[1]), c=in_x_label, s=100, alpha=0.5)
plt.show()
