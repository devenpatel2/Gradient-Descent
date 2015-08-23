#!/usr/bin/env python
#Linear regression using BFGS (gradient descent)
import numpy as np
from scipy.optimize import minimize
    
def costFunction(theta,x,y):
    ''' Computes the cost function '''
    x_o=np.ones((x.shape[0],1))

    N=len(y)
    X=np.zeros((x.shape[0],x.shape[1]+1))
    X[:,0:1]=x_o
    X[:,1:]=x
    cf=np.sum(np.power(np.dot(X,theta)-y,2))/(2*N)
    return cf


def costFunction_der(theta,x,y):
    ''' computes the derivative of the cost function'''

    N=len(y)
    x_o=np.ones((x.shape[0]))

    X=np.zeros((x.shape[0],x.shape[1]+1))
    X[:,0]=x_o
    X[:,1:]=x
    
    err=np.dot(X,theta)-y
    grad=np.dot(X.T,err)/N
    return grad
    
def minGD(data):
    initial=np.zeros(data.shape[1])
    x=data[:,0:-1]
    y=data[:,-1]
    res=minimize(costFunction,initial,args=(x,y),method='BFGS',jac=costFunction_der,options={'disp':True})
    return res

def predict(theta,x):
    #x=np.asarray(x)
    if x.ndim==1:
        x=x.reshape([1,len(x)])

    x_o=np.ones((x.shape[0]))
    X=np.zeros((x.shape[0],x.shape[1]+1))
    X[:,0]=x_o
    X[:,1:]=x
    return np.dot(X,theta)

if __name__=="__main__":
    import os
    import sys
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("-t","--train",required=False,help="data file for training")
    ap.add_argument("-p","--predict",nargs='+',required=False,help="input for prediction")
    ap.add_argument("-i","--input",required=False,help="file for input values for prediction")
    args=vars(ap.parse_args())

    if args["train"]!=None: 
        data=np.loadtxt(args["train"],delimiter=",")
        res=minGD(data)
        print "Parameters computed :"
        print res.x
        np.savetxt('params.txt',res.x)
        print "Prameters stored in 'params.txt'"
    
    if args["predict"]!=None:
        input_val=args["predict"]
        if os.path.isfile("params.txt"):
            theta=np.loadtxt("params.txt")
            X=[float(x) for x in   input_val]
            if len(X)+1 ==len(theta):
                print predict(theta,np.asarray(X))
            else:
                print "input value and the parameter size don't match"
                sys.exit(0)
        else:
            print "'params.txt' file missing. Train the data first to get the training parameters"
            sys.exit(0)
    
    if args["input"]!=None:
        input_val=np.loadtxt(args["input"],delimiter=',')
        if os.path.isfile("params.txt"):
            theta=np.loadtxt("params.txt")
            X=input_val
            if X.ndim==1:
                X=X.reshape([len(X),1])
            if len(X.T)+1 ==len(theta):
                print predict(theta,np.asarray(X))
            else:
                print "input value and the parameter size don't match"
                sys.exit(0)
 
        else:
            print "'params.txt' file missing. Train the data first to get the training parameters"
            sys.exit(0)

