

#from activationNmos import activationNmos
import Fpaa_reservoir as fpaa
import numpy as np
import scipy.io
import time


######################### Parameters #########################
units = 400
sp_rad = 0.8
inputScaling = 1
leakingRate = 1
sparsity = 0.02
mask = [0.49, 0.51]


##################### Dataset Parameters #####################

mat = scipy.io.loadmat('cochlea_test.mat')
X_train = mat['tmp']
X_train = X_train[:, 1:]
np.random.seed(0)
np.random.shuffle(np.transpose(X_train))


y_train = np.zeros((10, 2543))
m = 0
L = [250, 255, 255, 254, 255, 254, 254, 256, 256, 254]
for k in range(10):
    for i in range(L[k]):
        y_train[k][i+m] = 1
    m += L[k] 
np.random.seed(0)
np.random.shuffle(np.transpose(y_train))


(r,c) = y_train.shape
print("Training on " + str(c) + " data.")


############### Initialization of the Objects ################
relu = np.vectorize(lambda x: np.maximum(0,x))
sigmoid = np.vectorize(lambda x: 1/(1+np.exp(-x)))


reservoir = fpaa.Reservoir_FPAA(units = units,
                                inputs = 78,
                                sp_rad = sp_rad,
                                inputScaling = inputScaling,
                                leakingRate = leakingRate,
                                activation = relu,
                                sparsity=sparsity,
                                mask=mask)


########################## Training ##########################
reservoir.initializeState(X_train[:reservoir.inputs, 0])
print("...")
start = time.time()
reservoir.fit(X_train, y_train)
end = time.time()
print("Training done ! It took " + str(round((end - start)/60, 1)) + "min")
reservoir.resetState()
reservoir.initializeState(X_train[:reservoir.inputs, 0])

#reservoir.details()
reservoir.saveModel()

########################### Testing ##########################

mat = scipy.io.loadmat('cochlea_train.mat')
X_test = mat['tmp']
X_test = X_test[:, 1:]    
np.random.seed(0)
np.random.shuffle(np.transpose(X_test))    


y_test = np.zeros((10, 1593))
m = 0
L = [159, 159, 160, 159, 160, 159, 159, 160, 160, 158]
for k in range(10):
    for i in range(L[k]):
        y_test[k][i+m] = 1
    m += L[k] 
np.random.seed(0)
np.random.shuffle(np.transpose(y_test))
    


(r,c) = y_test.shape
print("Testing on " + str(c) + " data.")
print('...')

start = time.time()
predicted_results = []
for k in range(c):
    y = reservoir.predict(X_test[:,k])
    predicted_results.append(y)
end = time.time()
print("Prediction done in " + str(round((end - start)/60, 1)) + "min")
   

#################### Analyzing the results ###################
real_results = []
for k in range(c):
    real_results.append(np.argmax(y_test[:, k]))
    
correct = 0
erreur = 0

for k in range(len(real_results)):
    if predicted_results[k] == real_results[k]:
        correct += 1
    else :
        erreur += 1

print("Nombre de réponses corrects : " + str(correct))
print("Nombre d'erreurs faites : " + str(erreur))
print("La précision est donc de " + str(correct/(correct + erreur)*100) + "%.")

