import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


L = pd.read_csv('csv_activation_BenHug.csv')
x = L.iloc[:, 0].values
y = L.iloc[:, 1].values * 10**6


def f(z):
    f = scipy.interpolate.interp1d(x,y,kind='nearest',fill_value="extrapolate")
    return f(z)


def activationOTA(x):
    return(np.vectorize(f)(x))


x_ = np.linspace(0.7, 1.3, 100)
y_ = activationOTA(x_)


def test():
    plt.title("Activation function of Benji&Hugo")
    plt.plot(x_, y_, 'r.-')
    plt.plot(x, y, 'g--')
    plt.grid()
    plt.legend(['polynomial approximation', 'ground truth'])
    plt.show()
    
#test
    


