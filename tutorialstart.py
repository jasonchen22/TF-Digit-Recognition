

import tensorflow as tf

# initialize variable as 0
state = tf.Variable(0, name="counter")

# create a operation to add 1 to state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# create init operation
init_op = tf.initialize_all_variables()

# run operation
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# Another example
# finding factors by Gradient Descent
import numpy as np
x_data = np.float32(np.random.rand(2, 100))
# W = [0.1, 0.2], b = 0.3
y_data = np.dot([0.1, 0.2], x_data) + 0.3

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer  = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(0, 301):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))






