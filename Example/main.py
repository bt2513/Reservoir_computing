# Testing the trained reservoir at the Digital Spoken recognition task
# This reservoir had an accuracy of 98.7% during the test on 1593 data. 


import Fpaa_reservoir as fpaa
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


############### Initialization of the reservoir ################
relu = np.vectorize(lambda x: max(0,x))
sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))


reservoir = fpaa.Reservoir_FPAA(units = 400,
                                inputs = 78,
                                sp_rad = 0.8,
                                activation = sigmoid,
                                noise = True)

#reservoir.details()

reservoir.loadModel('Reservoir_poidsEntree.csv',
                    'Reservoir_poidsInternes.csv',
                    'Reservoir_poidsSortie.csv')

reservoir.resetState()


########################### Testing ##########################
mat = scipy.io.loadmat('five.mat')
#mat = scipy.io.loadmat('six.mat')
#mat = scipy.io.loadmat('zero.mat')
X_test = mat['coch']


reservoir.initializeState(X_test)


result, mean, result2d = reservoir.predict(X_test)
plt.imshow(mean)
plt.title('"Probabilities" calculated by the reservoir')
plt.colorbar()
print("Résultat prédit : " + str(result))
