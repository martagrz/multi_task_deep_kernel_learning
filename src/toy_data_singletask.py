import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

from dkl import DKL
from utils import load_dataset, func_task_one, func_task_two, rmse, add_column

#Load dataset
n = 100 
x_range = [-10,10]
y1, x1, x1_tst, y1_tst, x1_s, y1_s = load_dataset(func=func_task_one,n=n,n_tst=n,x_range=x_range)
y2, x2, x2_tst, y2_tst, x2_s, y2_s = load_dataset(func=func_task_two,n=n,x_range=[-10,-3,3,10],augment_x=True)

plt.scatter(x1,y1, label="sample")
plt.plot(x1_s,y1_s, label="ground truth")
plt.legend()
plt.title("Dataset 1")
plt.show()

plt.scatter(x2,y2, label="sample")
plt.plot(x2_s,y2_s, label="ground truth")
plt.legend()
plt.title("Dataset 2")
plt.show()

x1_new = add_column(x1, 0)
x2_new = add_column(x2, 1)
x = np.vstack((x1_new,x2_new))

y1_new = y1[:,None]
y2_new = y2[:,None]
y = np.concatenate((y1,y2))[:,np.newaxis]

model = DKL(x1_new,y1_new, sigma_sq=0.0001, learn_rate = 1e-3)
try:
    for i in range(1000):
        nlml = model.train_step()
        print("Epoch", i, "loss:", nlml)
except KeyboardInterrupt:
    pass

x1_s_new = add_column(x1_s, 0)
pred, sd = model.predict(x1_s_new)
rmse_1 = rmse(np.array(pred)[:,0], np.array(y1_s))
plt.scatter(x1,y1,marker='x',c='black')
plt.plot(x1_s,y1_s,c='red',label='ground truth')
plt.plot(x1_s,pred[:,0],c='blue',label='model')
plt.fill_between(x1_s,pred[:,0]-sd,pred[:,0]+sd,alpha=0.5, label='confidence region')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset 1")
plt.savefig('../figures/toy_data_dkl_single_one.png')
plt.show()

# Train model for task two
model = DKL(x2_new,y2_new, sigma_sq=0.0001, learn_rate = 1e-3)
try:
    for i in range(1000):
        nlml = model.train_step()
        print("Epoch", i, "loss:", nlml)
except KeyboardInterrupt:
    pass  

x2_s_new = add_column(x2_s, 1)
pred, sd = model.predict(x2_s_new)
rmse_2 = rmse(np.array(pred)[:,0], np.array(y2_s))
plt.scatter(x2,y2,marker='x',c='black')
plt.plot(x2_s,y2_s,c='red',label='ground truth')
plt.plot(x2_s,pred[:,0],c='blue',label='model')
plt.fill_between(x2_s,pred[:,0]-sd,pred[:,0]+sd,alpha=0.5, label='confidence region')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Dataset 2")
plt.savefig('../figures/toy_data_dkl_single_two.png')
plt.show()


print('RMSE:')
print('--------------------')
print('Task one:', rmse_1)
print('Task two:', rmse_2)  

