import tensorflow as tf
import copy
import random as rd
import numpy as np
import time
import loss

from tensorflow.examples.tutorials.mnist import input_data

batch_size = 50

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def re_x_size(x_size,strides,padding):
    s = strides[1:3]
    if padding == 'VALID':
        x_size = np.divide(x_size,s)
    else:
        for i in range(2):
            if x_size[i] % s[i] ==0:
                x_size[i] = x_size[i] / s[i]
            else:
                x_size[i] = x_size[i] / s[i] + 1
    return x_size

def conv2d(inputs,num_filters_in,num_filters_out,weights_size,num_parameter,strides=[1,1,1,1],padding='SAME'):
    weights_shape = [weights_size[0],weights_size[1],
                     num_filters_in,num_filters_out]
    weights = weight_variable(weights_shape)
    bias_shape = [num_filters_out]
    biases = bias_variable(bias_shape)
    conv = tf.nn.conv2d(inputs, weights, strides=strides,padding=padding)
    output = tf.nn.bias_add(conv, biases)
    num_parameter += num_filters_in * num_filters_out * weights_size[0] * weights_size[1]
    return output,num_parameter

def max_pool(inputs, k_size=[1,2,2,1], strides = [1,2,2,1], padding='SAME'):
    return tf.nn.max_pool(inputs,ksize=k_size, strides = strides, padding=padding)

def avg_pool(inputs, k_size=[1,2,2,1],strides=[1,2,2,1],padding='SAME'):
    return tf.nn.avg_pool(inputs,ksize=k_size,strides=strides,padding=padding)

def conv1_layer(inputs,num_filters_in,x_size,num_parameter):
    if num_filters_in == 1:
        num_filters_out = 16
    else:
        num_filters_out = num_filters_in
    h,num_parameter = conv2d(inputs,num_filters_in,num_filters_out,[3,3],num_parameter)
    output = tf.nn.relu(h)
    x_size = re_x_size(x_size,[1,1,1,1],padding='SAME')
    return output,x_size,num_filters_out,num_parameter

def conv2_layer(inputs,num_filters_in,x_size,num_parameter):
    if num_filters_in == 1:
        num_filters_out = 16
    else:
        num_filters_out = 16 + num_filters_in
    h,num_parameter = conv2d(inputs,num_filters_in,num_filters_out,[3,3],num_parameter,[1,2,2,1])
    output = tf.nn.relu(h)
    x_size = re_x_size(x_size,[1,2,2,1],padding='SAME')
    return output,x_size,num_filters_out,num_parameter

def pool1_layer(inputs,num_filters_in,x_size,num_parameter):
    outputs = max_pool(inputs)
    x_size = re_x_size(x_size,[1,2,2,1],padding='SAME')
    return outputs,x_size,num_filters_in,num_parameter

def pool2_layer(inputs,num_filters_in,x_size,num_parameter):
    outputs = avg_pool(inputs)
    x_size = re_x_size(x_size,[1,2,2,1],padding='SAME')
    return outputs,x_size,num_filters_in,num_parameter

def inception1_layer(inputs,num_filters_in,x_size,num_parameter):
    if num_filters_in == 1:
        num_filters_out = 16
    else:
        num_filters_out = num_filters_in
    branch11,num_parameter = conv2d(inputs,num_filters_in,num_filters_out,[1,1],num_parameter)
    branch33,num_parameter = conv2d(inputs,num_filters_in,num_filters_out/4*3,[1,1],num_parameter)
    branch33,num_parameter = conv2d(branch33,num_filters_out/4*3,num_filters_out,[3,3],num_parameter)
    branch_pool = avg_pool(inputs,[1,2,2,1],[1,1,1,1])
    branch_pool,num_parameter = conv2d(branch_pool,num_filters_in,num_filters_out,[1,1],num_parameter)
    output = tf.concat(3,[branch11,branch33,branch_pool])
    return output,x_size,3*num_filters_out,num_parameter

def inception2_layer(inputs,num_filters_in,x_size,num_parameter):
    if num_filters_in == 1:
        num_filters_out = 16
    else:
        num_filters_out = num_filters_in + 16
    branch11,num_parameter = conv2d(inputs,num_filters_in,num_filters_out,[1,1],num_parameter)
    branch33,num_parameter = conv2d(inputs,num_filters_in,num_filters_out/4*3,[1,1],num_parameter)
    branch33,num_parameter = conv2d(branch33,num_filters_out/4*3,num_filters_out,[1,3],num_parameter)
    branch33,num_parameter = conv2d(branch33,num_filters_out,num_filters_out,[3,1],num_parameter)
    branch_pool = avg_pool(inputs,[1,2,2,1],[1,1,1,1])
    branch_pool,num_parameter = conv2d(branch_pool,num_filters_in,num_filters_out,[1,1],num_parameter)
    output = tf.concat(3,[branch11,branch33,branch_pool])
    return output,x_size,3*num_filters_out,num_parameter

def output(h,x_size,depth,keep_prob,num_parameter):
    x_dim=x_size[0] * x_size[1] * depth
    W = weight_variable([x_dim,1024])
    b = bias_variable([1024])
    num_parameter += x_dim * 1024 + 1024
    h_flat = tf.reshape(h, [-1, x_dim])
    h1 = tf.nn.relu(tf.matmul(h_flat, W) + b)
    #drop out.
    h_drop = tf.nn.dropout(h1, keep_prob)
    
    W2 = weight_variable([1024, 10])
    b2 = bias_variable([10])
    num_parameter += 1024*10 + 10
    logits = tf.matmul(h_drop, W2) + b2
    return logits,num_parameter

def can_train(x_size,i):
    if i == 6 and x_size[0] >=5 and x_size[1]>=5:
        return 1
    if (i == 5 or i <= 2) and x_size[0] >=3 and x_size[1] >=3:
        return 1
    if (i == 3 or i == 4) and x_size[0] >=2 and x_size[1] >=2:
        return 1
    return 0

layer_f={1:conv1_layer, 2:conv2_layer, 3:pool1_layer, 4:pool2_layer, 5:inception1_layer,6:inception2_layer}
loss_f={1:loss.loss1, 2:loss.loss2, 3:loss.loss3, 4:loss.loss4, 5:loss.loss5, 6:loss.mean_squared_error,
    7:loss.cross_entropy, 8:loss.hellinger_dis}
x = tf.placeholder(tf.float32, shape=[None, 28*28])
labels = tf.placeholder(tf.float32, shape=[None,10])

dnn = list()
lost = list()
dnn.append([1,3,1,2])
for j in range(1,20):
    k = list()
    for i in range(4):
        k.append(rd.randint(1,4))
    dnn.append(copy.deepcopy(k))
print dnn

x_image = tf.reshape(x, [-1,28,28,1])
sess = tf.InteractiveSession()

for k in range(20):
    print("DNN%d"%k)
    print dnn[k]

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    nottrain=0
    h = x_image
    x_size = [28,28]
    channel = 1
    num_parameter = 0
    for i in dnn[k]:
        if can_train(x_size,i) == 1:
            h,x_size,channel,num_parameter = layer_f[i](h,channel,x_size,num_parameter)
        else:
            nottrain = 1

    if nottrain == 1:
        print "not train"
        lost.append([])
        continue
    print x_size
    keep_prob = tf.placeholder(tf.float32)
    logits,num_parameter = output(h,x_size,channel,keep_prob,num_parameter)

    sess.run(tf.initialize_all_variables())
    c = loss.r_square(logits,labels)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(c)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    duration = 0
    acc=list()
    for i in range(1,1000):
        batch = mnist.train.next_batch(batch_size)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                                           x:batch[0], labels: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            acc.append(train_accuracy*100)
        start_time = time.time()
        train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})
        duration += time.time() - start_time
    variance=np.var(acc)
    test_accuracy = accuracy.eval(feed_dict={
                             x: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0})
    print("%g"%test_accuracy)
    lost.append([test_accuracy,duration,num_parameter,variance])
    print lost[k]
