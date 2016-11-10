import tensorflow as tf
import ga1 as ga
import loss
import layer
import copy
import train

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 28*28])
labels = tf.placeholder(tf.float32, shape=[None,10])

dnn = list()
loss_sequence = list()
loss_sequence = ga.initialisation(loss_sequence,10,5,5)

x_image = tf.reshape(x, [-1,28,28,1])
config = tf.ConfigProto()
config.gpu_option.allow_growth = True
sess = tf.Session(config=config)


for le in range(20):
    dnn_init = list()
    loss_fitness = list()
    dnn_init.append([1,3,1,2])
    dnn_init = ga.initialisation(dnn_init,5,4,4)
    losslen = len(loss_sequence)
    for lstep in range(losslen):
        print("loss_se",lstep,loss_sequence[lstep])
        dnn = copy.deepcopy(dnn_init)
        loss_fitness = train.dnn_evolve_train(dnn,mnist,x_image,loss_sequence[lstep],loss_fitness,
                                              x,labels,sess,5)
        print("loss_fitness",loss_fitness[lstep])
    if le != 20:
        loss_sequence = ga.loss_add(loss_sequence,loss_fitness,10,5)
