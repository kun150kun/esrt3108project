import random as rd
import numpy as np

def soft_max(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 0)

def selection(dnn_population,lost,rank):
    x = soft_max(lost)
    d = rd.random()
    for i in range(len(dnn_population)):
        d = d - x[i][rank]
        if d <= 0:
            break
    return i

def mutation(dnn):
    l = len(dnn)
    pos = rd.randint(0,l-1)
    while(1):
        s = rd.randint(1,6)
        if(s != dnn[pos]):
            dnn[pos] = s
            break
    return dnn

def reproduce(dnn1,dnn2):
    l = len(dnn1)
    pos = rd.randint(0,l-1)
    print dnn1,dnn2,pos
    return  dnn1[:pos] + dnn2[pos:]

def initialisation(dnn_population,num = 10,len = 5):
    for i in range(num):
        k = list()
        for j in range(len):
            k.append(rd.randint(1,6))
        dnn_population.append(k)

def add(dnn_population,rank,add_num = 4):
    l = len(dnn_population)
    if l < 2:
        initialisation(dnn_population,10-l)
    for i in range(add_num):
        x = selection(dnn_population,lost,rank)
        y = x
        while(y==x):
            y = selection(dnn_population,lost,rank)
        child = reproduce(dnn_population[x],dnn_population[y])
        p = rd.randint(0,10000)
        if p ==1024:
            child = mutation(child)
        dnn_population.append(child)


if __name__ == '__main__':
    dnn_population = list()
    initialisation(dnn_population)
    lost = list()
    initialisation(lost,10,2)
    print dnn_population
    print lost
    add(dnn_population,1)
    print dnn_population
