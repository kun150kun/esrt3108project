import random as rd
import numpy as np
import tensorflow as tf
import copy

def soft_max(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 0)

def dnn_selection(dnn_population,lost):
    t = copy.deepcopy(lost)
    for i in range(len(t)):
        t[i][1] = -5 * t[i][1]
    lo = np.exp(t) / np.sum(np.exp(t),axis = 0)
    d = rd.random()
    for i in range(len(dnn_population)):
        d = d - lo[i][1]
        if d <= 0:
            break
    return i

def loss_selection(loss_population,lost):
    t = copy.deepcopy(lost)
    print t
    for i in range(len(t)):
        t[i][0] = t[i][0] * 8
    lo = np.exp(t) / np.sum(np.exp(t),axis = 0)
    d = rd.random()
    print "lost",lo
    for i in range(len(loss_population)):
        d = d - lo[i][0]
        if d<=0:
            break
    return i

def mutation(dnn,num_layer):
    l = len(dnn)
    pos = rd.randint(0,l-1)
    while(1):
        s = rd.randint(1,num_layer)
        if(s != dnn[pos]):
            dnn[pos] = s
            break
    return dnn

def reproduce(dnn1,dnn2):
    l = len(dnn1)
    pos = rd.randint(1,l-1)
    return  dnn1[:pos] + dnn2[pos:]

def initialisation(dnn_population,num = 10,len = 5,num_layer = 6):
    for i in range(num):
        k = list()
        for j in range(len):
            k.append(rd.randint(1,num_layer))
        dnn_population.append(k)
    return dnn_population

def dnn_add(dnn_population,fitness,add_num = 4,num_layer = 4):
    l = len(dnn_population)
    if l < 2:
        initialisation(dnn_population,10-l)
    for i in range(add_num):
        x = dnn_selection(dnn_population,fitness)
        y = x
        while(y==x):
            y = dnn_selection(dnn_population,fitness)
        child = reproduce(dnn_population[x],dnn_population[y])
        p = rd.randint(0,100)
        if p == 32:
            child = mutation(child,num_layer)
        dnn_population.append(child)
    return dnn_population

def loss_add(loss_population,fitness,add_num = 4,num_layer = 4):
    l = len(loss_population)
    for i in range(add_num):
        x = loss_selection(loss_population,fitness)
        y = x
        while(y == x):
            y = loss_selection(loss_population,fitness)
        child = reproduce(loss_population[x],loss_population[y])
        p = rd.randint(0,100)
        if p == 32:
            child = mutation(child,num_layer)
        loss_population.append(child)
    return loss_population

if __name__ == '__main__':
    dnn_population = list()
    initialisation(dnn_population)
    lost = list()
    initialisation(lost,10,2)
    print dnn_population
    print lost
    add(dnn_population,1)
    print dnn_population
