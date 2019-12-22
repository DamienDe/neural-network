import numpy as np
import abc

class IActivation():
    @abc.abstractmethod
    def activation_forward(self, Z, **kwargs):
        pass

    @abc.abstractmethod
    def activation_backward(self, dA, **kwargs):
        pass


class IReluActivation(IActivation):
    def activation_forward(self, Z, **kwargs):
        A = np.maximum(0,Z)
        return A
    
    def activation_backward(self, dA, **kwargs):
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.        
        # When z <= 0, you should set dz to 0 as well. 
        dZ[self.Z <= 0] = 0
        return dZ

class ISigmoidActivation(IActivation):
    def activation_forward(self, Z, **kwargs):
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def activation_backward(self, dA, **kwargs):
        s = self.activation_forward(self.Z)
        dZ = dA * s * (1 - s)
        return dZ