import numpy as np
import matplotlib.pyplot as plt
import csv

global V

V = []

with open('csv_activation.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
        L = []
        for value in row:
            L.append(float(value))
        V.append(L)

    
def activation(x, vr=0.8, a=2.0,s=1.0):
    global V
    fi = min(max(100.0*(vr+1)/11,0.0),99.0)
    if not fi.is_integer():
        i = int(fi)
        f = V[i]
    else:
        f = V[int(fi)]
    
    fx = min(max(1000.0*(x+5)*s,0.0),24999.0)
    if not fx.is_integer():
        x = int(fx)
        return a*((x+1-fx)*(f[x]-f[x+1])+f[x+1])
    else:
        return a*f[int(fx)]
    
    
def activationNmos(M):
    return(np.vectorize(activation)(M))
    
def test():
    x = np.linspace(0, 10, 10000)
    y = activationNmos(x)
    plt.plot(x,y,'r')
    plt.title("Activation function")
    plt.grid()
    plt.show()
    
    
test()

