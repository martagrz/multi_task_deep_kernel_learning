import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from dkl import DKL
from utils import load_dataset, func_task_one, func_task_two, rmse

#Load dataset
n = 100 
x_range = [-10,10]
y1, x1, x1_tst, y1_tst, x1_s, y1_s = load_dataset(func=func_task_one,n=n,x_range=x_range)
y2, x2, x2_tst, y2_tst, x2_s, y2_s = load_dataset(func=func_task_two,n=n,x_range=[-10,-3,3,10],augment_x=True)

plt.scatter(x1,y1)
plt.plot(x1_s,y1_s)
plt.show()

#Model fit 1

model = DKL(x1[:,None], y1[:,None],kernel='rbf')
for i in range(500): 
	nlml = model.train_step()
	print("Epoch", i, "loss:", nlml)

pred_x, sd_x = model. predict(x1_s[:,np.newaxis])

plt.scatter(x1,y1,marker='x',c='black')
plt.plot(x1_s,y1_s,c='red',label='ground truth')
plt.plot(x1_s,pred_x,c='blue',label='model')
plt.fill_between(x1_s,pred_x[:,0]-sd_x,pred_x[:,0]+sd_x,alpha=0.5, label='confidence region')
plt.legend()
plt.show()

rmse_1 = rmse(np.array(pred_x), np.array(y1_s))
print('RMSE:', rmse_1)  

