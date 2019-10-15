#python3 -m pip install -r requirements.txt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class DKL(tf.keras.models.Model):
    def __init__(self, X_train, y_train, sigma_sq, learn_rate=1e-4):
        super(DKL, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.y_dim = y_train.shape[1]
        self.X_dim = X_train.shape[1]
        self.N = len(y_train)
        assert self.N == len(X_train)

        self.sigma_sq = sigma_sq

        self.trans_dim = 1
        self.transformation_net = tf.keras.models.Sequential([      
        #tf.keras.layers.Dense(32*2, activation="tanh"),
        #tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(32, activation="tanh"),        
        tf.keras.layers.Dense(self.trans_dim, activation=None)])

        #self.length_scale = tf.Variable(1., dtype=tf.float64, name="length_scale")

        self.I = np.eye(self.N)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
    def transformation(self, X):
        return self.transformation_net(X)# * self.length_scale

    def distance_sq(self, A, B):
        diff = A[:,tf.newaxis,:] - B[tf.newaxis,:,:]
        diff_sq = tf.math.square(diff)
        return tf.reduce_sum(diff_sq, axis=-1)

    def get_kernel(self, A, B):
        A_trans = self.transformation(A)
        B_trans = self.transformation(B)
        distances_sq = self.distance_sq(A_trans, B_trans)

        kern_output = tf.math.exp(-distances_sq)

        return kern_output

    def get_nlml(self):
        K = self.get_kernel(self.X_train, self.X_train) + self.sigma_sq * self.I
        K_inv = tf.linalg.inv(K)

        model_fit = tf.transpose(self.y_train) @ K_inv @ self.y_train
        complexity = tf.linalg.logdet(K)
        nlml = tf.math.reduce_sum(model_fit) + complexity

        return nlml

    def predict(self, X_test): 
        #Need to make x_train and x_test go through all the transformations in the network 
        K_train_train = self.get_kernel(self.X_train, self.X_train) + self.sigma_sq * self.I
        K_test_train = self.get_kernel(X_test, self.X_train)
        K_test_test = self.get_kernel(X_test, X_test)
        K_inv = tf.linalg.inv(K_train_train)

        E_f = K_test_train @ K_inv @ self.y_train
        cov_f = K_test_test - K_test_train @ K_inv @ (tf.transpose(K_test_train))
        stddev = tf.math.sqrt(tf.linalg.diag_part(cov_f))

        return E_f, stddev

    def train_step(self):
        with tf.GradientTape() as tape:
            nlml = self.get_nlml()

        gradients = tape.gradient(nlml, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return nlml.numpy()


if __name__ == "__main__":
    x = 5 * np.linspace(-1, 1, 10).astype(np.float64)
    y = np.sin(x) / x

    plt.plot(x,y, "o")
    plt.show()

    model = DKL(x[:,np.newaxis], y[:,np.newaxis], sigma_sq=0.0001)
    try:
        for i in range(2000):
            nlml = model.train_step()
            print("Epoch", i, "loss:", nlml)
    except KeyboardInterrupt:
        pass

    x_test = 5 * np.linspace(-1, 1, 100).astype(np.float64)
    pred_x, sd_x = model.predict(x_test[:,np.newaxis])

    plt.scatter(x,y)
    plt.plot(x_test,pred_x)
    plt.show()

    plt.scatter(x,y)
    plt.plot(x_test,pred_x[:,0])
    plt.fill_between(x_test,pred_x[:,0]-sd_x,pred_x[:,0]+sd_x,alpha=0.5)
    plt.show()

