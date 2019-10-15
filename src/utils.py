
import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(1)

def add_column(x, value):
    return np.stack((x, value * np.ones(len(x)))).T

def load_dataset(func, n=100, n_tst=100,x_range = [-10,10], noise=True, sigma=0.001, augment_x=False):
	'''
	This function generates a training dataset of size n, a test datset of size n_tst, 
	over a range of x inputs x_range, which is a list of two numbers in ascending order. 
	- func is the function governing the output, which includes the input x and the random noise. 
	- sigma is the true variance of the white noise process.  
	'''
	np.random.seed(43)
	#Generate training dataset
	if augment_x: 
		x = np.concatenate((np.random.uniform(x_range[0],x_range[1],n//2),np.random.uniform(x_range[2],x_range[3],n//2)))
	else: 
		x = np.random.uniform(x_range[0],x_range[1],n)

	if noise: 
		eps = np.random.randn(n) * sigma
		y = func(x) + eps 
	else: 
		y = func(x)

	#Generate test dataset 
	x_tst = np.linspace(x_range[0],x_range[-1], num=n_tst).astype(np.float32)

	if noise: 
		eps_tst = np.random.randn(n_tst) * sigma
		y_tst = func(x_tst) + eps_tst
	else: 
		y_tst = func(x_tst)

	#Ground truth for the correct specified 
	xs = np.linspace(x_range[0], x_range[-1], n_tst)
	ys = func(xs)

	return y, x, x_tst, y_tst, xs, ys

#Credit to Boustati for these two functions to illustrate the multi-task problem
def func_task_one(x): 
	y = np.sinc(x)
	return y
  
def func_task_two(x):
	y = np.sinc(x) + np.sinc(5-x) + np.sinc(5+x)
	return y 

def func_task_transpose(x):
	return func_task_one(x-4)


def plot_figures(x1,y1,x1_s,y1_s,x2,y2,x2_s,y2_s): 
	plt.figure(figsize=(16,9))
	plt.subplot(221)
	plt.plot(x1_s, y1_s)
	plt.scatter(x1, y1, c='orange', marker='x')

	plt.subplot(222)
	plt.plot(x2_s, y2_s)
	plt.scatter(x2, y2, c='orange', marker='x')

	plt.show()
    
def load_stepfunc_dataset(n=100, sigma=0.1, noise=False):
    eps = np.zeros(n) 
    if noise: 
        eps = np.random.randn(n) * sigma 
    
    eps = np.array(eps)[:,None]
    
    x_step = np.array(np.append(np.random.uniform(0,5,n//2),np.random.uniform(5,10,n//2)))[:,None] 
    y_step = np.array(np.append(np.repeat(-0.5,n//2),np.repeat(0.5,n//2)))[:, None] + eps
    x_line = np.arange(0,10,0.01)[:,None]
    y_line = np.where(x_line>5,0.5,-0.5)
    
    return x_step, y_step, x_line, y_line


def rmse(predictions, targets):
    return np.sqrt(((predictions.flatten() - targets.flatten()) ** 2).mean())

