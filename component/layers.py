from interface.activation import *
from interface.hiddenlayer import *

class SimpleReluLayer(ISimpleHiddenLayer,IReluActivation):
    def __init__(self, number_input_unit, number_hidden_unit):
        super().__init__(number_input_unit, number_hidden_unit) 

class SimpleReluLayerRegularized(ISimpleHiddenLayer, IReluActivation):
    def __init__(self, number_input_unit, number_hidden_unit, lambd):
        super().__init__(number_input_unit, number_hidden_unit) 
        self.lambd = lambd

    def linear_backward(self, dZ, **kwargs):
        dA_prev, dW, db = super().linear_backward(dZ, **kwargs)       
        dW = dW + (self.lambd/self.A_prev.shape[1])*self.W
        return dA_prev, dW, db

    def additionnal_cost(self):
        return (self.lambd/(2*self.A_prev.shape[1])) * np.sum(np.square(self.W))

class NormalizedReluLayer(INormalizedHiddenLayer, IReluActivation):
    def __init__(self, number_input_unit, number_hidden_unit, coef_beta):
        super().__init__(number_input_unit, number_hidden_unit, coef_beta) 

class NormalizedReluLayerRegularized(INormalizedHiddenLayer, IReluActivation):
    def __init__(self, number_input_unit, number_hidden_unit, lambd, coef_beta):
        super().__init__(number_input_unit, number_hidden_unit, coef_beta) 
        self.lambd = lambd

    def linear_backward(self, dZ, **kwargs):
        dA_prev, dW, dgamma, dbeta  = super().linear_backward(dZ, **kwargs)     
        dW = dW + (self.lambd/self.A_prev.shape[1])*self.W
        return dA_prev, dW, dgamma, dbeta

    def additionnal_cost(self):
        return (self.lambd/(2*self.A_prev.shape[1])) * np.sum(np.square(self.W))  

class SimpleSigmoidLayer(ISimpleHiddenLayer, ISigmoidActivation):
    def __init__(self, number_input_unit, number_hidden_unit):
        super().__init__(number_input_unit, number_hidden_unit)

class SimpleSigmoidLayerRegularized(SimpleSigmoidLayer):
    def __init__(self, number_input_unit, number_hidden_unit, lambd):
        super().__init__(number_input_unit, number_hidden_unit) 
        self.lambd = lambd

    def linear_backward(self, dZ, **kwargs):
        dA_prev, dW, db = super().linear_backward(dZ, **kwargs)    
        dW = dW + (self.lambd/self.A_prev.shape[1])*self.W
        return dA_prev, dW, db

    def additionnal_cost(self):
        return (self.lambd/(2*self.A_prev.shape[1])) * np.sum(np.square(self.W))

class NormalizedSigmoidLayer(INormalizedHiddenLayer, ISigmoidActivation):
    def __init__(self, number_input_unit, number_hidden_unit, coef_beta):
        super().__init__(number_input_unit, number_hidden_unit, coef_beta)