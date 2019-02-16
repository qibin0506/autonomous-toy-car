import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import numpy as np
import csv
import model

lr = 0.0001
batch_size = 400
epochs = 200
batch_start = 0
l2_hyper_param = 0.001

data = []
train_data = []
validation_data = []

with open("./data/record.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        data.append(line)

train_data = shuffle(data)

batch_count = int(len(train_data) / batch_size)
print(batch_count)


def next_batch():
    global batch_start

    images = []
    angles = []

    end = batch_start+batch_size
    if end + batch_size > len(train_data):
        end = len(train_data)

    for batch_sample in train_data[batch_start:end]:
        # if float(batch_sample[0]) < 0.0:
        #     continue

        angle = float(batch_sample[0])

        name = './data/img/' + batch_sample[1].split('/')[-1]
        raw_image = cv2.imread(name)

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        image = image[80:]
        image = image / 255.0 - 0.5

        angle = float(angle) / 180.0 - 0.5

        images.append(image)
        angles.append([angle])

    batch_start = batch_start + batch_size

    return shuffle(np.array(images), np.array(angles))


train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(model.result, model.tf_y)))\
        + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * l2_hyper_param
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y = [], []

    for epoch in range(epochs):
        train_data = shuffle(train_data)

        batch_start = 0
        cur_cost = 0
        print("start epoch " + str(epoch + 1) + "/" + str(epochs))

        for i in range(batch_count):
            x, y = next_batch()
            _, batch_cost = sess.run([optimizer, loss], feed_dict={model.tf_x: x, model.tf_y: y, model.keep_prob: 0.8})
            cur_cost += batch_cost / batch_count

        print("loss: ", cur_cost)
        if epoch == 50 or epoch == 100 or epoch == 150:
            print("save model: ./model/tf_result_" + str(epoch) + ".ckpt")
            saver.save(sess, "./model/tf_result_" + str(epoch) + ".ckpt")

    saver.save(sess, "./model/tf_result.ckpt")
