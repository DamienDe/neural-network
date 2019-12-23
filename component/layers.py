from interface.activation import *
from interface.hiddenlayer import *
import numpy as np

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

class SimpleReluLayerWithDropOut(ISimpleHiddenLayer,IReluActivation):
    def __init__(self, number_input_unit, number_hidden_unit, keep_prob):
        super().__init__(number_input_unit, number_hidden_unit)
        self.keep_prob = keep_prob
    def forward_propagation(self, A_prev, prediction=False, checking=False, **kwargs):
        A = super().forward_propagation(A_prev, **kwargs)
        if (not prediction):
            if (not checking):
                #Setting drop out
                self.drop_mask = np.random.rand(A.shape[0],A.shape[1]) < self.keep_prob
            A = np.multiply(A,self.drop_mask)
            A /= self.keep_prob
        return A

    def back_propagation(self, dA, prediction=False, checking=False, **kwargs):
        if (not prediction):
            #Setting drop out
            dA = np.multiply(dA,self.drop_mask)
            dA /= self.keep_prob
        dA_prev = super().back_propagation(dA, **kwargs)
        return dA_prev


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