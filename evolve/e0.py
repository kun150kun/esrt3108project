import tensorflow as tf
import ga1 as ga
import random as rd
import numpy as np
import time
import loss
import layer
import copy
import train

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 50
def can_train(x_size,i):
    if i == 6 and x_size[0] >=5 and x_size[1]>=5:
        return 1
    if (i == 5 or i <= 2) and x_size[0] >=3 and x_size[1] >=3:
        return 1
    if (i == 3 or i == 4) and x_size[0] >=2 and x_size[1] >=2:
        return 1
    return 0

x = tf.placeholder(tf.float32, shape=[None, 28*28])
labels = tf.placeholder(tf.float32, shape=[None,10])

dnn = list()
loss_sequence = list()
loss_sequence = ga.initialisation(loss_sequence,10,5,5)

x_image = tf.reshape(x, [-1,28,28,1])
sess = tf.Session()

losslen = len(loss_sequence)
for le in range(20):
    dnn_init = list()
    loss_fitness = list()
    dnn_init.append([1,3,1,2])
    dnn_init = ga.initialisation(dnn_init,5,4,4)
    for lstep in range(losslen):
        print(loss_sequence[lstep])
        dnn = copy.deepcopy(dnn_init)
        loss_fitness = train.dnn_evolve_train(dnn,mnist,x_image,loss_sequence[lstep],loss_fitness,5)
        print("loss_fitness",loss_fitness[lstep])
    if le != 20:
        loss_sequence = ga.loss_add(loss_sequence,loss_fitness,3,6)
