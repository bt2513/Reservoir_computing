import numpy as np
import matplotlib.pyplot as plt


class Reservoir_FPAA():

    def __init__(self, units, sp_rad, activation, output = 10, inputScaling = 1, leakingRate = 1,
                 sparsity = 0.2, weights = None, noise = False):
        """
            Implementation of a reservoir computer for the fpaa board
            It is specifically designed for the digit recognition task 
            
            Attributes
            ----------
            units (int):
                Number of nodes in the reservoir
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
        """
        self.units = units
        self.sp_rad = sp_rad
        self.inputScaling = inputScaling
        self.leakingRate = leakingRate
        self.sparsity = sparsity
        self.activation = np.vectorize(activation)
        self.weights = weights
        self.W, self.Win,_ = self.initWeights()
        self.Wout = np.matrix(np.zeros((output, self.units + 2)))
        self.X_state = np.matrix(np.zeros((self.units, 1)))
        self.output = output
        self.noise = noise

    
    def initWeights(self):
        """
            Initializes the weight matrix W, the weights of the input Win 
            
            Returns 
            -------
            (W, Win, eigenMax)
        """
        if self.weights != None:
            pass
            # To be completed. Depends on the capacities we can use on the FPAA board
        else:
            W = np.matrix([[np.random.rand()-0.5 if np.random.rand() < self.sparsity else 0 for j in range(self.units)] for i in range(self.units)])
            Win = np.matrix(np.random.rand(self.units, 2)-0.5) * self.inputScaling
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
            
    
    def readOutput(self, u):
        """
            Reads the output of the matrix for an input u
            
            Parameters
            ----------
            u (float):
                scalar input
                
            Returns
            -------
            y_k (array):
                vector of the output
        """
        y_k = np.dot(self.Wout, np.concatenate((np.matrix(np.array([[1], [u]])), self.X_state), axis = 0))
        return(y_k)
    
    
    def updateState(self, u):
        """
            Updates the state matrix of the reservoir for the input u
            
            Parameters
            ----------
            u (float):
                scalar input
        """
        bruit = 0
        tmp = np.dot(self.Win, np.matrix(np.array([[1], [u]]))) + np.dot(self.W, self.X_state)
        self.X_state = (1-self.leakingRate) * self.X_state + self.leakingRate * np.matrix(self.activation(tmp))
        if self.noise :
            bruit = np.matrix(0.001 * np.random.normal(0, 1, self.units)).reshape(-1, 1)
        return(self.X_state + bruit)
        
        
    def initializeState(self, U):
        """
            Initializes the state matrix of the reservoir for the input U
            
            Parameters
            ----------
            u (list):
                column vector of the input
        """
        U = U.reshape(1, -1)
        (un, T) = U.shape
        for k in range(T):
            self.X_state = self.updateState(U[0][k])
            
    
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
        """
        U = U.reshape(1, -1)
        (un, T) = U.shape
        if self.output > 1:
            tmp = np.matrix(np.zeros((10, 1)))
            for k in range(T):
                self.updateState(U[0][k])
                y_k = self.readOutput(U[0][k])
                tmp += y_k
            y_mean = (1/T)*tmp
            #return(y_mean)
            return(np.argmax(y_mean))
        else:
            y = []
            for k in range(T):
                self.updateState(U[0][k])
                y_k = float(self.readOutput(U[0][k]))
                y.append(y_k)
            return(np.array(y))
        
    
    def stateNodes(self, U):
        """
            Plots the states of the reservoir's nodes
        
            Parameters
            ----------
            U (list):
                column vector of the input
                
            Returns
            -------
            states (array):
                matrix composed of the state values of the reservoir's nodes.
                Each row stands for one node.
        """
        U = U.reshape(1, -1)
        (un, T) = U.shape
        states = np.zeros((self.units, T))       
        for k in range(T): 
            state = self.updateState(U[0][k])
            for n in range(self.units):
                states[n, k] = state[n, 0]
        plt.figure()
        plt.title("State of the nodes")
        for n in range(self.units):
            plt.plot(states[n,:],label='node' + str(n))
        plt.legend()
        plt.show()
        return(states)
        
        
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
        if self.output == 1:
            Y = Y.reshape(1, -1)
            U = U.reshape(1, -1)
            (un, T) = U.shape
            first = True
            X_states = np.zeros((self.units + 2, 1)) # matrice de vecteurs d'états
            for i in range(T):
                state = self.updateState(U[0][i])
                tmp = np.concatenate((np.matrix(np.array([[1], [U[0][i]]])), state), axis = 0)
                if first:
                    X_states = tmp
                    first = False
                else :
                    X_states = np.concatenate((X_states, tmp), axis = 1)              
            self.Wout = np.dot(Y, np.linalg.pinv(X_states))    # Moindres carrés
            y_training = np.array(np.dot(self.Wout, X_states))
            return(np.sqrt(((y_training - np.array(Y))**2).mean())/np.var(y_training))
        
        else :
            (y, t) = Y.shape
            (u, T) = U.shape
            first = True
            X_states = np.zeros((self.units + 2, 1)) # matrice de vecteurs d'états
            for i in range(T):  # construction du vecteur d'etats pour chaque entrée 
                mean = np.zeros((self.units + 2, 1))
                for k in range(u):
                    state = self.updateState(U[k][i])
                    tmp1 = np.concatenate((np.matrix(np.array([[1], [U[k][i]]])), state), axis = 0)
                    mean += tmp1
                mean = (1/u) * mean                  
                if first:
                    X_states = mean
                    first = False
                else :
                    X_states = np.concatenate((X_states, mean), axis = 1)                  
            B = np.linalg.inv(np.dot(X_states, np.transpose(X_states)))
            if np.linalg.det(B) > 0:
                A = np.dot(Y, np.transpose(X_states))
                self.Wout = np.matrix(np.dot(A, B))
            else:
                self.Wout = np.matrix(np.dot(Y, np.linalg.pinv(X_states)))     # Moindres carrés
            
            
            
            