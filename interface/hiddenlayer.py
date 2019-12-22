import abc
import numpy as np

class IHiddenLayer:
    position = 0
    def __init__(self, **kwargs):
        IHiddenLayer.position +=1
        self.position = IHiddenLayer.position
        pass

    def forward_propagation(self, A_prev, **kwargs):
        #Linear part
        Z = self.linear_forward(A_prev, **kwargs)
        #Activation part
        A = self.activation_forward(Z, **kwargs)
        return A

    def back_propagation(self, dA, **kwargs):
        #Activation part
        dZ = self.activation_backward(dA, **kwargs)
        #Linear part     
        parameters_var = self.linear_backward(dZ, **kwargs)  
        #Saving parameter variation
        self.parameters_var = parameters_var
        #Returning dA
        return parameters_var[0]

    @abc.abstractmethod
    def linear_forward(self, **kwargs):
        pass

    @abc.abstractmethod
    def linear_backward(self, dZ, **kwargs):
        pass

    @abc.abstractmethod
    def update_parameters(self, learning_rate, **kwargs):
        pass

    @abc.abstractmethod
    def extract_gradient(self):
        pass

    @abc.abstractmethod
    def extract_parameters(self):
        pass
    
    def additionnal_cost(self):
        return 0

class ISimpleHiddenLayer(IHiddenLayer):
    def __init__(self, number_input_unit, number_hidden_unit):
        super().__init__()
        self.W = np.random.randn(number_hidden_unit,number_input_unit)/ np.sqrt(number_input_unit)
        self.b = np.zeros((number_hidden_unit,1))

    def linear_forward(self, A_prev, **kwargs):
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if("W" + str(self.position) in value):
                    self.W = value["W" + str(self.position)]
                if("b" + str(self.position) in value):
                    self.b = value["b" + str(self.position)]
        Z = np.dot(self.W,A_prev) + self.b
        #Saving data
        self.Z = Z
        self.A_prev = A_prev
        return Z

    def linear_backward(self, dZ, **kwargs):
        m = self.A_prev.shape[1]        
        dW = 1./m * np.dot(dZ, self.A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T,dZ)
        return dA_prev, dW, db   

    def update_parameters(self, learning_rate, **kwargs):
        self.W = self.W - learning_rate * self.parameters_var[1]
        self.b = self.b - learning_rate * self.parameters_var[2]

    def extract_gradient(self):
        grads = {}
        grads["dW" + str(self.position)], grads["db" + str(self.position)] = self.parameters_var[1], self.parameters_var[2]
        return grads

    def extract_parameters(self):
        params = {}
        params["W" + str(self.position)], params["b" + str(self.position)] = self.W, self.b
        return params

class INormalizedHiddenLayer(IHiddenLayer):
    def __init__(self, number_input_unit, number_hidden_unit, coef_beta):
        super().__init__()
        self.W = np.random.randn(number_hidden_unit,number_input_unit)/ np.sqrt(number_input_unit)
        self.gamma = np.random.randn(number_hidden_unit,1)/ np.sqrt(number_input_unit)
        self.beta = np.zeros((number_hidden_unit,1))
        self.coef_beta = coef_beta

        #Estimate weight average
        self.V_mu = np.zeros((number_hidden_unit,1))
        self.V_sigma_square_plus_e = np.zeros((number_hidden_unit,1))

    def linear_forward(self, A_prev, checking=False, prediction=False, **kwargs):
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if("W" + str(self.position) in value):
                    self.W = value["W" + str(self.position)]
                if("gamma" + str(self.position) in value):
                    self.gamma = value["gamma" + str(self.position)]
                if("beta" + str(self.position) in value):
                    self.beta = value["beta" + str(self.position)]

        Z = np.dot(self.W,A_prev)
        N = Z.shape[1]

        e = 10e-8

        if prediction :
            mu = self.V_mu
            sigma_square_plus_e = self.V_sigma_square_plus_e
        else:
            mu = 1./N * np.sum(Z,axis=1, keepdims=True)
            sigma_square_plus_e = 1./N * np.sum((Z - mu)**2, axis=1, keepdims=True) + e
            
        #Computing esimate weight average
        if(not checking):   
            self.V_mu = (self.coef_beta*self.V_mu + (1- self.coef_beta)*mu)
            self.V_sigma_square_plus_e = (self.coef_beta*self.V_sigma_square_plus_e + (1- self.coef_beta)*sigma_square_plus_e)

        Z_minus_mu = Z - mu

        Z_norm = Z_minus_mu / (np.sqrt(sigma_square_plus_e))
        
        Z_tilde = Z_norm * self.gamma + self.beta

        #Saving data
        self.Z = Z_tilde
        self.Z_norm = Z_norm
        self.A_prev = A_prev
        self.Z_minus_mu = Z_minus_mu
        self.sigma_square_plus_e = sigma_square_plus_e
        return Z_tilde
        
    def linear_backward(self, dZ_tilde, **kwargs):
        
        N,m = dZ_tilde.shape

        #step 9
        dbeta = 1./ m * np.sum(dZ_tilde, axis=1, keepdims=True)
        dgamma_z_norm = dZ_tilde

        #step 8
        dgamma = 1./ m * np.sum(dgamma_z_norm * self.Z_norm, axis=1, keepdims=True)
        dz_norm = dgamma_z_norm * self.gamma

        #step 7
        dz_minus_mu_1 = dz_norm / np.sqrt(self.sigma_square_plus_e)
        dinv_sqrt_sigma_square_plus_e = 1./ m * np.sum(dz_norm*self.Z_minus_mu,axis=1, keepdims=True)

        #step 6
        dsqrt_sigma_square_plus_e = dinv_sqrt_sigma_square_plus_e * (-1. / np.sqrt(self.sigma_square_plus_e)**2)

        #step 5
        dvar = dsqrt_sigma_square_plus_e * 0.5 * (1. / np.sqrt(self.sigma_square_plus_e))

        #step 4
        dZ_minus_mu_square = dvar * np.ones(dZ_tilde.shape)

        #step3
        dZ_minus_mu_2 = dZ_minus_mu_square * 2 * self.Z_minus_mu

        #step2
        dmu = np.sum(dz_minus_mu_1 + dZ_minus_mu_2, axis=1, keepdims=True) * -1.
        dz1 = (dz_minus_mu_1 + dZ_minus_mu_2) * 1.

        #step1
        dz2 = 1./ m * dmu * np.ones(dZ_tilde.shape)

        #step0
        dZ = dz1 + dz2

        dW = 1./m * np.dot(dZ, self.A_prev.T)

        dA_prev = np.dot(self.W.T,dZ)

        return dA_prev, dW, dgamma, dbeta 

    def update_parameters(self, learning_rate, **kwargs):
        self.W = self.W - learning_rate * self.parameters_var[1]
        self.gamma = self.gamma - learning_rate * self.parameters_var[2]
        self.beta = self.beta - learning_rate * self.parameters_var[3]

    def extract_gradient(self):
        grads = {}
        grads["dW" + str(self.position)], grads["dgamma" + str(self.position)], grads["dbeta" + str(self.position)] = self.parameters_var[1], self.parameters_var[2], self.parameters_var[3]
        return grads

    def extract_parameters(self):
        params = {}
        params["W" + str(self.position)], params["gamma" + str(self.position)] , params["beta" + str(self.position)]= self.W, self.gamma, self.beta
        return params