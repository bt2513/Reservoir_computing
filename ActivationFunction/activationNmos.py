import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


L = pd.read_csv('csv_activation_CorJo.csv')
x = L.iloc[1:2480, 0].values
y = L.iloc[1:2480, 1].values


def f(z):
    f = scipy.interpolate.interp1d(x,y,kind='nearest',fill_value="extrapolate")
    return f(z)


def activationNMOS(x):
    return(np.vectorize(f)(x))


x_ = np.linspace(0, 5, 100)
y_ = activationNMOS(x_)


def test():
    plt.title("Activation function of Coco&Jo")
    plt.plot(x_, y_, 'r.-')
    plt.plot(x, y, 'g--')
    plt.grid()
    plt.legend(['polynomial approximation', 'ground truth'])
    plt.show()
    
#test()



