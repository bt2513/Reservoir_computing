# Plot all the activation function used during our project


import activationNMOS as nmos
import activationOTA as ota
import numpy as np
import matplotlib.pyplot as plt


# Activation function of Corentin & Joseph
nmos.test()

# Activation function of Benjamin & Hugo
ota.test()

# relu
relu = np.vectorize(lambda x: np.maximum(0,x))
x = np.linspace(-5, 5, 1000)
y_ = relu(x)
plt.plot(x, y_, 'r-')
plt.grid()
plt.title('ReLu')
plt.show()

# sigmoid
sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))
y__ = sigmoid(x)
plt.plot(x, y__, 'g-')
plt.grid()
plt.title('Sigmoid')
plt.show()

# tanh
tanh = np.vectorize(np.tanh)
y___ = tanh(x)
plt.plot(x, y___, 'b-')
plt.grid()
plt.title('Tanh')
plt.show()

# Plot all
x = np.linspace(-5, 5, 100)
y_ = relu(x)
y__ = sigmoid(x)
y___ = tanh(x)
y1 = nmos.activationNMOS(x)
y2 = ota.activationOTA(x)
plt.plot(x, y_, 'r-', label='relu')
plt.plot(x, y__, 'g-', label = 'sigmoid')
plt.plot(x, y___, 'b-', label='tanh')
plt.plot(x, y1, 'm-', label='activationNmos')
plt.plot(x, y2, 'c-', label='activationOTA')
plt.grid()
plt.legend()
plt.title('Activation function used')
plt.show()






