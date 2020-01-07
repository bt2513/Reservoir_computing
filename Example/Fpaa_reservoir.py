import numpy as np
import pandas as pd


class Reservoir_FPAA():

    def __init__(self, units, inputs, sp_rad, activation, output = 10, inputScaling = 1, leakingRate = 1,
                 sparsity = 0.1, weights = None, noise = False, mask = None):
        """
            Implementation of a reservoir computer for the fpaa board
            It is specifically designed for the digit recognition task 
            
            Attributes
            ----------
            units (int):
                Number of nodes in the reservoir
            inputs (int):
                Dimension of the input
            output (int):
                Number of class in the output
            sp_rad (float): 
                spectral radius of the Weight matrix
            leakingRate (float):
                Leaking rate of the reservoir
            sparsity (float):
                Between 0 and 1. Sparsity of the connections in the reservoir
            weights (array):
                list of the different capacities available we can use on the fpaa
            X_state (array):
                Matrix of the state of the nodes
            mask (array):
                The mask can only have two values [x1, x2]. Those values are chosen 
                randomly when initializing the Weight matrix W (when the component is != 0)
        """
        self.units = units
        self.inputs = inputs
        self.sp_rad = sp_rad
        self.inputScaling = inputScaling
        self.leakingRate = leakingRate
        self.sparsity = sparsity
        self.activation = np.vectorize(activation)
        self.weights = weights
        self.mask = mask
        self.W, self.Win, self.rayon_spec = self.initWeights()
        self.Wout = np.matrix(np.zeros((output, self.units + self.inputs + 1)))
        self.X_state = np.matrix(np.zeros((self.units, 1)))
        self.output = output
        self.noise = noise  

    
    def initWeights(self):
        """
            Initializes the weight matrix W (according to the mask and the sparsity),
            and the input weights Win randomly.
            
            Returns 
            -------
            (W, Win, eigenMax)
        """
        if self.weights != None:
            pass
            # To be completed. Depends on the capacities we can use on the FPAA board
        else:
            if self.mask != None:
                np.random.seed(0)
                W = np.zeros((self.units, self.units))
                for i in range(self.units):
                    for j in range(self.units):
                        if np.random.rand() < self.sparsity:
                            if np.random.rand() < 0.5:
                                W[i, j] = self.mask[0]
                            else :
                                W[i, j] = self.mask[1]
                W = np.matrix(W)
                Win = np.matrix(np.random.rand(self.units, 1 + self.inputs)-0.5) * self.inputScaling
                eigen,_ = np.linalg.eig(W)
                return(W, Win, max(eigen.real))
            else:
                np.random.seed(0)
                W = np.matrix([[np.random.rand()-0.5 if np.random.rand() < self.sparsity else 0 for j in range(self.units)] for i in range(self.units)])
                Win = np.matrix(np.random.rand(self.units, 1 + self.inputs)-0.5) * self.inputScaling
                eigen,_ = np.linalg.eig(W)
                W = W * self.sp_rad/max(eigen.real)
                eigen,_ = np.linalg.eig(W)
                return(W, Win, max(eigen.real))
        
    
    def details(self):
        """
            Prints the characteristics of the reservoir
        """
        print("output : " + str(self.output) + ", " + str(type(self.output)) + "\n")
        print("W : " + str(self.W.shape) + ", " +  str(type(self.W)) + "\n")
        print("Win : " + str(self.Win.shape) + ", " + str(type(self.Win)) + "\n")
        print("Wout : " + str(self.Wout.shape) + ", " + str(type(self.Wout)) + "\n")
        print("X_state : " + str(self.X_state.shape) + ", " + str(type(self.X_state)))
        
    
    def resetState(self, rand = False):
        """
            Resets the state matrix of the reservoir (X_state)
            
            Parameters
            ----------
            rand (boolean):
                if True, the state of the nodes are initialized randomly. 
                Else, they are initialized at 0.
        """
        if rand:
            self.X_state = np.matrix(np.random.rand(self.units, 1))
        else:
            self.X_state = np.matrix(np.zeros((self.units, 1)))
            
    
    def readOutput(self, U):
        """
            Reads the output of the matrix for an input U
            
            Parameters
            ----------
            U (vector column):
                input
                
            Returns
            -------
            y_k (array):
                vector of the output
        """
        U = U.reshape(-1, 1)
        U = np.matrix(U)
        tmp = np.concatenate((np.matrix([1]), U), axis = 0)
        X = np.concatenate((tmp, self.X_state), axis = 0)
        y_k = np.dot(self.Wout, X)
        return(y_k)
    
    
    def updateState(self, U):
        """
            Updates the state matrix of the reservoir for the input U
            
            Parameters
            ----------
            U (vector column):
                input
        """
        bruit = 0
        U = U.reshape(-1, 1)
        U = np.matrix(U)
        X = np.concatenate((np.matrix([1]), U), axis = 0)
        tmp = np.dot(self.Win, X) + np.dot(self.W, self.X_state)
        self.X_state = (1-self.leakingRate) * self.X_state + self.leakingRate * np.matrix(self.activation(tmp))
        if self.noise :
            bruit = np.matrix(0.001 * np.random.normal(0, 1, self.units)).reshape(-1, 1)
        return(self.X_state + bruit)
        
        
    def initializeState(self, M):
        """
            Initializes the state matrix of the reservoir for the input U
            
            Parameters
            ----------
            M (matrix):
                column vector of the input
        """
        #M = M.reshape((-1, 1), order='F')
        #M = np.matrix(M)
        #self.X_state = self.updateState(M)
        self.predict(M)
            
    
    def predict(self, U): 
        """
            Classifies the input U
            
            Parameters
            ----------
            U :
                column vector of the input
                
            Returns
            -------
            class (int):
                class of U
            y_mean (array):
                vector output of the mean probabilities of the classes
            result2d (array2d):
                output that can be printed to observe the results
        """
        U = U.reshape(-1, 1)
        U = np.matrix(U)
        (V, un) = U.shape
        tmp = np.matrix(np.zeros((10, 1)))
        result2d = np.matrix(np.zeros((10, 1)))
        first = True
        for j in range(int(V/self.inputs)):
            a = self.inputs * j
            b = self.inputs * (j+1)
            column = U[a:b, 0].reshape(-1, 1)
            self.updateState(column)
            y_k = self.readOutput(column)
            if first :
                result2d = y_k
                first = False
            else :
                result2d = np.concatenate((result2d, y_k), axis = 1)
            tmp += y_k
        y_mean = (1/(int(V/self.inputs)))*tmp
        #return(y_mean)
        return(np.argmax(y_mean), y_mean, result2d)
        
        
    def fit(self,U,Y):
        """
            Trains the output weights of the reservoir 
            
            Parameters
            ----------
            U (array):
                array of the training input ((len(U), number of samples) or (1, len(u)))
            Y (array):
                array of the target output ((len(Y), number of samples) or (1, len(y)))
        """
        (y, t) = Y.shape
        (V, T) = U.shape
        first = True
        X_states = np.zeros((self.units + self.inputs + 1, 1)) # matrice de vecteurs d'états
        for i in range(T):  # construction du vecteur d'etats pour chaque entrée 
            mean = np.zeros((self.units + self.inputs + 1, 1))
            for j in range(int(V/self.inputs)):
                a = self.inputs * j
                b = self.inputs * (j+1)
                column = U[a:b, i].reshape(-1, 1)
                state = self.updateState(column)
                column = np.matrix(column)
                Z = np.concatenate((np.matrix([1]), column), axis = 0)
                tmp1 = np.concatenate((Z, state), axis = 0)
                mean += tmp1
            mean = (1/(int(V/self.inputs))) * mean                  
            if first:
                X_states = mean
                first = False
            else :
                X_states = np.concatenate((X_states, mean), axis = 1)                  
        self.Wout = np.matrix(np.dot(Y, np.linalg.pinv(X_states)))     # Moindres carrés
        
    
    def saveModel(self, fileprefix = 'Reservoir_'):
        """
            Save the weights of the reservoir
            
            Parameters
            ----------
            fileprefix (string):
                prefix of the csv filenames storing the weights of the reservoir
        """
        np.savetxt(fileprefix+"poidsInternes.csv", self.W, delimiter=",")
        np.savetxt(fileprefix+"poidsSortie.csv", self.Wout, delimiter=",")
        np.savetxt(fileprefix+"poidsEntree.csv", self.Win, delimiter=",")

    
    def loadModel(self, Win, W, Wout):
        """
            Load the trained weights given in arguments.
            
            Parameters
            ----------
            Win (csv file):
                Input weights of the reservoir
            W (csv file):
                Internal weights of the reservoir
            Wout (csv file):
                Output weights of the reservoir
        """
        self.Win = np.matrix(pd.read_csv(Win, header=None))
        self.W = np.matrix(pd.read_csv(W, header=None))
        self.Wout = np.matrix(pd.read_csv(Wout, header=None))

        
            
            
            
            