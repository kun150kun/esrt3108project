import tensorflow as tf
import numpy as py

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

layer_f={1:conv1_layer, 2:conv2_layer, 3:pool1_layer, 4:pool2_layer, 5:inception1_layer,6:inception2_layer}
