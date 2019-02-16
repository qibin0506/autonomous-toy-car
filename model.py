import tensorflow as tf

image_width = 320
image_height = 160

tf_x = tf.placeholder(tf.float32, shape=[None, image_height, image_width, 3])
tf_y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

mu = 0.0
sigma = 0.1

conv1_W = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 24], mean=mu, stddev=sigma))
conv1_b = tf.Variable(tf.constant(0.1, shape=[24]))
conv1 = tf.nn.conv2d(input=tf_x, filter=conv1_W, strides=[1, 2, 2, 1], padding='VALID') + conv1_b
conv1 = tf.nn.relu(conv1)

conv2_W = tf.Variable(tf.truncated_normal(shape=[5, 5, 24, 36], mean=mu, stddev=sigma))
conv2_b = tf.Variable(tf.constant(0.1, shape=[36]))
conv2 = tf.nn.conv2d(input=conv1, filter=conv2_W, strides=[1, 2, 2, 1], padding='VALID') + conv2_b
conv2 = tf.nn.relu(conv2)

conv3_W = tf.Variable(tf.truncated_normal(shape=[5, 5, 36, 48], mean=mu, stddev=sigma))
conv3_b = tf.Variable(tf.constant(0.1, shape=[48]))
conv3 = tf.nn.conv2d(input=conv2, filter=conv3_W, strides=[1, 2, 2, 1], padding='VALID') + conv3_b
conv3 = tf.nn.relu(conv3)

conv4_W = tf.Variable(tf.truncated_normal(shape=[3, 3, 48, 64], mean=mu, stddev=sigma))
conv4_b = tf.Variable(tf.constant(0.1, shape=[64]))
conv4 = tf.nn.conv2d(input=conv3, filter=conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
conv4 = tf.nn.relu(conv4)

conv5_W = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=mu, stddev=sigma))
conv5_b = tf.Variable(tf.constant(0.1, shape=[64]))
conv5 = tf.nn.conv2d(input=conv4, filter=conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
conv5 = tf.nn.relu(conv5)

dropout = tf.nn.dropout(conv5, keep_prob=keep_prob)

dropout_shape = dropout.get_shape().as_list()
flatten_size = dropout_shape[1] * dropout_shape[2] * dropout_shape[3]
flatten = tf.reshape(dropout, [-1, flatten_size])

fc1_W = tf.Variable(tf.truncated_normal(shape=[flatten.get_shape().as_list()[1], 100], mean=mu, stddev=sigma))
fc1_b = tf.Variable(tf.constant(0.1, shape=[100]))
fc1 = tf.matmul(flatten, fc1_W) + fc1_b
fc1 = tf.nn.relu(fc1)

fc2_W = tf.Variable(tf.truncated_normal(shape=[100, 50], mean=mu, stddev=sigma))
fc2_b = tf.Variable(tf.constant(0.1, shape=[50]))
fc2 = tf.matmul(fc1, fc2_W) + fc2_b
fc2 = tf.nn.relu(fc2)

fc3_W = tf.Variable(tf.truncated_normal(shape=[50, 10], mean=mu, stddev=sigma))
fc3_b = tf.Variable(tf.constant(0.1, shape=[10]))
fc3 = tf.matmul(fc2, fc3_W) + fc3_b
fc3 = tf.nn.relu(fc3)

fc4_W = tf.Variable(tf.truncated_normal(shape=[10, 1], mean=mu, stddev=sigma))
fc4_b = tf.Variable(tf.constant(0.1, shape=[1]))
result = tf.matmul(fc3, fc4_W) + fc4_b