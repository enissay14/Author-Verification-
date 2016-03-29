import numpy as np

#f_Mtrain = open('Mtrain',"r") 
Mtrain = np.load('Mtrain.npy')

#f_target = open('Target',"r") 
target = np.load('Target.npy')

Mtrain.tofile('Mtrain.csv',sep=',')
target.tofile('target.csv',sep=',')