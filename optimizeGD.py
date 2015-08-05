#!/usr/bin/env python
#Linear regression using BFGS (gradient descent)
import numpy as np
from scipy.optimize import minimize
    
def costFunction(theta,x,y):
''' Computes the cost function '''
    x_o=np.ones((x.shape[0],1))

    N=x.shape[0]
    X=np.zeros((x.shape[0],x.shape[1]+1))
    X[:,0:1]=x_o
    X[:,1:]=x
    #J=np.zeros([iterations,1])
    #grad=np.zeros(theta.shape)
    
    cf=np.sum(np.power(np.dot(X,theta)-y,2))/(2*N)
    #grad=np.dot(X.T,err)/N
    return cf


def costFunction_der(theta,x,y):
''' computes the derivative of the cost function'''

    N=x.shape[0]
    x_o=np.ones((x.shape[0],1))

    X=np.zeros((x.shape[0],x.shape[1]+1))
    X[:,0:1]=x_o
    X[:,1:]=x
    #J=np.zeros([iterations,1])
    #grad=np.zeros(theta.shape)
    
    err=np.dot(X,theta)-y
    grad=np.dot(X.T,err)/N
    return grad
    
def minGD(data,initial):
    x=data[:,0:-1]
    y=data[:,-1]
    res=minimize(costFunction,initial,args=(x,y),method='BFGS',jac=costFunction_der,options={'disp':True})
    return res

if __name__=="__main__":
    import sys
    data=np.loadtxt(sys.argv[1],delimiter=",")
    initial=np.zeros(data.shape[1])
    res=minGD(data,initial)
    print "Theta computed :"
    print res.x
