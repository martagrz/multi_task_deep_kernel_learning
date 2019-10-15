import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from dkl import DKL
from utils import rmse

sarcos_train = scipy.io.loadmat('../data/sarcos_inv.mat')
train_data = sarcos_train['sarcos_inv']
sarcos_test = scipy.io.loadmat('../data/sarcos_inv_test.mat')
test_data = sarcos_test['sarcos_inv_test']

n = 1000

x_train = train_data[:n,:21]
y_train = train_data[:n,21:]

plt.scatter(x_train[:,0],y_train[:,0])
plt.show()

x_test = test_data[:n,:21]
y_test = test_data[:n,21:]

model = DKL(x_train,y_train, sigma_sq=1e-3, learn_rate=0.001)
try:
    for i in range(50000):
        nlml = model.train_step()
        print("Epoch", i, "loss:", nlml)
except KeyboardInterrupt:
    pass


pred, sd = model.predict(x_test)
rmse = rmse(np.array(pred),np.array(y_test))

plt.scatter(x_train[:,0],y_train[:,0])
plt.scatter(x_test[:,0],pred[:,0])
plt.show()
print('RMSE:', rmse)


