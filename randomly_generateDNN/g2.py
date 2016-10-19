import tensorflow as tf
import copy
import random as rd

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def re_x_size(x_size,strides,padding):
    s = strides[1:3]
    if padding == 'VALID':
        x_size = np.divide(x_size,strides)
    else:
        for i in range(2):
            if x_size[i] % strides[i] ==0:
                x_size[i] = x_size[i] / s[i]
            else:
                x_size[i] = x_size[i] / s[i] + 1
    return x_size

def re_w_b_shape(w_shape):
    w1_shape = copy.deepcopy(w_shape)
    w1_shape[2] = w_shape[3]
    w1_shape[3] = w_shape[3] * 2
    b1_shape = [w1_shape[3]]
    return w1_shape,b1_shape

def output(h,w1,b1,keep_prob):
    W = weight_variable(w1)
    b = bias_variable([w1[1]])

    h_flat = tf.reshape(h, [-1, w1[0]])
    h1 = tf.nn.relu(tf.matmul(h_flat, W) + b)

    h_drop = tf.nn.dropout(h1, keep_prob)

    W2 = weight_variable([w1[1], b1[0]])
    b2 = bias_variable(b1)

    logits = tf.matmul(h_drop, W2) + b2
    return logits

def conv1_layer(h,w_shape,b_shape,x_size):
    weight = weight_variable(w_shape)
    biases = bias_variable(b_shape)
    strides = [1,2,2,1]
    padding = 'SAME'
    
    h_c = tf.nn.relu(tf.nn.conv2d(h, weight, strides=strides,padding=padding) + biases)
    
    w1_shape,b1_shape = re_w_b_shape(w_shape)
    re_x_size(x_size,strides,padding)
    
    return h_c,w1_shape,b1_shape,x_size

def conv2_layer(h,w_shape,b_shape,x_size):
    weight = weight_variable(w_shape)
    biases = bias_variable(b_shape)
    strides = [1,1,1,1]
    padding = 'SAME'
    h_c = tf.nn.relu(tf.nn.conv2d(h, weight, strides=strides,padding=padding) + biases)
    
    w1_shape,b1_shape = re_w_b_shape(w_shape)
    x_size = re_x_size(x_size,strides,padding)
    
    return h_c,w1_shape,b1_shape,x_size


def pool1_layer(h,x_size):
    strides = [1,1,1,1]
    padding = 'SAME'
    h_p = tf.nn.avg_pool(h, ksize=[1,2,2,1], strides=strides,padding=padding)
    x_size = re_x_size(x_size,strides,padding)
    return h_p,x_size

def pool2_layer(h,x_size):
    strides = [1,2,2,1]
    padding = 'SAME'
    h_p = tf.nn.max_pool(h, ksize=[1,2,2,1], strides=strides,padding=padding)
    x_size = re_x_size(x_size,strides,padding)
    return h_p,x_size

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

layer_f={1:conv1_layer, 2:conv2_layer, 3:pool1_layer, 4:pool2_layer}

x = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None, 10])

dnn = list()
len = rd.randint(2,4)
k = list()
for i in range(len):
    k.append(rd.randint(1,4))
dnn.append(k)
print dnn

x_image = tf.reshape(x, [-1,28,28,1])
x_size = [28,28]
h = x_image
w_shape =[5,5,1,32]
b_shape =[32]
for i in dnn[0]:
    if i <=2:
        h,w_shape,b_shape,x_size=layer_f[i](h,w_shape,b_shape,x_size)
    else:
        h,x_size=layer_f[i](h,x_size)
print x_size
keep_prob = tf.placeholder(tf.float32)
w_o = [x_size[0]*x_size[1]*w_shape[2],1024]
b_o = [10]
logits = output(h,w_o,b_o,keep_prob)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                                        x:batch[0], labels: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
                                    x: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0}))
