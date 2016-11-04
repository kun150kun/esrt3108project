import tensorflow as tf
import math


def loss1(logits,labels):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,labels)
    return loss

def loss2(logits,labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits,labels)
    return loss

def loss3(logits,labels):
    p = tf.nn.softmax(logits)
    loss = tf.contrib.losses.cosine_distance(p,labels,0)
    return loss

def loss4(logits,labels):
    loss = tf.contrib.losses.log_loss(tf.nn.softmax(logits),labels)
    return loss

def loss5(logits,labels):
    loss = tf.contrib.losses.hinge_loss(tf.nn.softmax(logits),labels)
    return loss

def mean_squared_error(logits,labels):
    t = tf.sub(tf.nn.softmax(logits),labels)
    p = tf.square(t)
    loss = 0.5 * tf.reduce_mean(p,reduction_indices=[1])
    return loss

def cross_entropy(logits,labels):
    logits = tf.nn.softmax(logits)
    i = tf.constant(1,dtype = 'float32',shape = [10])
    t = labels * tf.log(logits) + tf.sub(i,labels) * tf.log(tf.sub(i,logits))
    loss = -tf.reduce_mean(t,reduction_indices=[1])
    return loss

def hellinger_dis(logits,labels):
    p = tf.nn.softmax(logits)
    t = tf.square(tf.sub(tf.sqrt(p),tf.sqrt(labels)))
    loss = 1 / math.sqrt(2) * tf.reduce_mean(t,reduction_indices=[1])
    return loss

#input cannot be zero.
def kullback_leibler_divergence(logits,labels):
    logits = tf.nn.softmax(logits)
    t = labels * tf.log(tf.truediv(labels,logits))
    loss = tf.reduce_mean(t,reduction_indices=[1])
    return loss

def r_square(logits,labels):
    logits = tf.nn.softmax(logits)
    y_ = tf.fill([10],tf.reduce_mean(labels))
    tot = tf.reduce_mean(tf.square(tf.sub(labels,y_)))
    res = tf.reduce_mean(tf.square(tf.sub(labels,logits)))
    loss = tf.truediv(res,tot)
    return loss
