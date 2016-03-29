import numpy as np

#f_Mtrain = open('Mtrain',"r") 
Mtrain = np.load('Mtrain.npy')

#f_target = open('Target',"r") 
target = np.load('Target.npy')

np.savetxt("Mtrain.csv", Mtrain, delimiter=",")
np.savetxt("target.csv", target, delimiter=",")