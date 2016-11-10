import tensorflow as tf
import ga1 as ga
import random as rd
import numpy as np
import time
import loss
import layer
import copy

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
    dnn_init = ga.initialisation(dnn_init,7,4,4)
    for lstep in range(losslen):
        print(loss_sequence[lstep])
        dnn = copy.deepcopy(dnn_init)
        dnn_fitness = list()
        lost = list()
        index = 0
        for i in range(10):
            dnnl = len(dnn)
            i0 = index
            for k in range(i0,dnnl):
                print("DNN%d"%k)
                print dnn[index]
                m0 = mnist
                nottrain=0
                h = x_image
                x_size = [28,28]
                channel = 1
                num_parameter = 0
                for j in dnn[k]:
                    if can_train(x_size,j) == 1:
                        h,x_size,channel,num_parameter = layer.layer_f[j](h,channel,x_size,num_parameter)
                    else:
                        nottrain = 1
                if nottrain == 1:
                    print "not train"
                    dnn = dnn[:index] + dnn[index + 1:]
                    continue
                print x_size
                keep_prob = tf.placeholder(tf.float32)
                logits,num_parameter = layer.output(h,x_size,channel,keep_prob,num_parameter)
                lll=list()
                for j in range(5):
                    lll.append(tf.reduce_mean(loss.loss_f[j+1](logits,labels)))
                correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
                train_step=list()
                for j in loss_sequence[lstep]:
                    train_step.append(tf.train.AdamOptimizer(1e-4).minimize(lll[j-1]))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                sess.run(tf.initialize_all_variables())
                duration = 0
                acc=list()
        
                for j in range(1001):
                    batch = m0.train.next_batch(batch_size)
                    if j%100 == 0:
                        train_accuracy = sess.run(accuracy,feed_dict={
                                                  x:batch[0], labels: batch[1], keep_prob: 1.0})
                        print(train_accuracy)
                        acc.append(train_accuracy*100)
            
                    start_time = time.time()
                    sess.run(train_step[index % 5],feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})
                    duration += time.time() - start_time
                variance=np.var(acc)
                test_accuracy = sess.run(accuracy,feed_dict={
                                            x: m0.test.images, labels: m0.test.labels, keep_prob: 1.0})
                lost.append(sess.run(lll,feed_dict={
                                        x: m0.test.images, labels: m0.test.labels, keep_prob: 1.0}))
                dnn_fitness.append([test_accuracy,duration,num_parameter,variance])
                print dnn_fitness[index],lost[index]
                index = index + 1
            if i != 9:
                dnn = ga.dnn_add(dnn,lost,3,4)
        loss_fitness.append([jj / len(dnn) for jj in np.sum(dnn_fitness,axis = 0)])
        print("loss_fitness",loss_fitness[lstep])
    if le != 20:
        loss_sequence = ga.loss_add(loss_sequence,loss_fitness,3,6)
