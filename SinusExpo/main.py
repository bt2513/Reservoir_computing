# Test of the Sinus to the power
# Assessment of the time series training of our reservoir


import numpy as np
import matplotlib.pyplot as plt
from activationNmos import activationNmos
import Fpaa_reservoir as fpaa


######################### Parameters #########################
N = 400     # Total number of data 
n = 240     # Number of data used to train the reservoir

units = 20
sp_rad = 0.8
inputScaling = 0.5
leakingRate = 1

############### Initialization of the Reservoir ################
reservoir = fpaa.Reservoir_FPAA(units = units,
                                sp_rad = sp_rad,
                                inputScaling = inputScaling,
                                leakingRate = leakingRate,
                                activation = activationNmos,
                                output = 1, 
                                noise = False,
                                sparsity = 0.2)


##################### Dataset Parameters #####################
k = np.linspace(1,N,N)/4
if reservoir.noise :
    b = 0.001 * np.random.normal(0, 1, len(k))
else:
    b=0
U = np.sin(k) + b   # input
Y = U**6    # output target - The power of the sinus can be changed here


########################## Training ##########################
reservoir.initializeState(U[:n])
print("Training" + "\n" + "...")
NRMSE_training = reservoir.fit(U[:n], Y[:n])
print("Training done !")
reservoir.resetState()
reservoir.initializeState(U[:n])

#reservoir.details()


########################### Testing ##########################
predicted_results = reservoir.predict(U[n:])


#################### Analyzing the results ###################
plt.title("Results")
plt.plot(predicted_results,'r.-')
plt.plot(Y[n:],'g--')
plt.legend(['predicted_results', 'ground truth'])
plt.show()

print("The training NRMSE is : " + str(round(NRMSE_training, 3)))

RMSE_testing = np.sqrt(((predicted_results - Y[n:])**2).mean())
NRMSE_testing = RMSE_testing / np.var(predicted_results)
print("The testing NRMSE is : " + str(round(NRMSE_testing, 3)))