import numpy as np
import matplotlib.pyplot as plt
import abc

class Network:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def train(self, X, Y, learning_rate=0.0075, nb_iterations=2000, gradient_check=False):
        costs = []
        for i in range(nb_iterations):
            #Computing forward activation
            AL = self.model_forward(X)

            if(i % 100 == 0):
                J = self.compute_cost(AL, Y)
                print(f'Cost after {i} iterations : {J}')
                costs.append(J)
            
            #Computing backward propagation
            grads = self.model_backward(AL, Y)

            #Checking gradient
            if(gradient_check):
                self.gradient_check_util(X, Y, grads)

            #Updating parameters
            self.update_parameters(learning_rate)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        #Retrieving parameters
        parameters = {}
        for l in (range(len(self.hidden_layers))):
            parameters.update(self.hidden_layers[l].extract_parameters())
        return parameters
    
    def predict(self, X, Y, **kwargs):
        m = X.shape[1]
        p = np.zeros((1,m))
        # Forward propagation
        AL = self.model_forward(X, **kwargs)
        # convert probas to 0/1 predictions
        for i in range(0, AL.shape[1]):
            if AL[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        print("Accuracy: "  + str(np.sum((p == Y)/m)))

    def update_parameters(self, learning_rate):
        for l in range(len(self.hidden_layers)):
            self.hidden_layers[l].update_parameters(learning_rate)
    
    def compute_cost(self, AL, Y):
        L = len(self.hidden_layers)
        m = Y.shape[1]
        additionnal_cost = 0
        for l in range(0, L):
            additionnal_cost = additionnal_cost + self.hidden_layers[l].additionnal_cost()
        cost = (-1/m) * np.sum((np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T))) + additionnal_cost
        return cost

    def model_forward(self, X, **kwargs):
        A = X
        #Computing forward activation
        for l in range(len(self.hidden_layers)):
            A = self.hidden_layers[l].forward_propagation(A, **kwargs)
        return A

    def model_backward(self, AL, Y): 
        grads = {}
        L = len(self.hidden_layers)
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dAL_prev = self.hidden_layers[L-1].back_propagation(dAL)
        
        # Computing rest of backpropagation
        for l in reversed(range(L-1)):
            dAL_prev = self.hidden_layers[l].back_propagation(dAL_prev)
        
        #Extracting gradients
        for l in (range(L)):
            grads.update(self.hidden_layers[l].extract_gradient())
            
        return grads

    def gradient_check_util(self, X, Y, gradients, epsilon = 1e-7):
        # Getting parameters
        parameters = {}
        L = len(self.hidden_layers)
        for l in (range(L)):
            parameters.update(self.hidden_layers[l].extract_parameters())

        # Set-up variables
        parameters_values, shapes = self.dictionary_to_vector(parameters)
        grad,shapes_grad = self.dictionary_to_vector(gradients)
        
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        m = X.shape[1]

        # Compute gradapprox
        for i in range(num_parameters):
            thetaplus = np.copy(parameters_values)                                     # Step 1
            thetaplus[i][0] = thetaplus[i][0] + epsilon                                # Step 2
            params = self.vector_to_dictionary(thetaplus,shapes)

            AL = self.model_forward(X, parameters=params, checking=True)
            J_plus[i] = self.compute_cost(AL, Y)

            thetaminus = np.copy(parameters_values)                                      # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon                               # Step 2   
            params = self.vector_to_dictionary(thetaminus,shapes)
            AL = self.model_forward(X, parameters=params, checking=True)
            J_minus[i] = self.compute_cost(AL, Y)

            gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

        approx_grad = self.vector_to_dictionary(gradapprox,shapes_grad)
        real_grad = self.vector_to_dictionary(grad,shapes_grad)

        numerator = np.linalg.norm(grad - gradapprox)                                           # Step 1'
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                                         # Step 2'
        difference = numerator / denominator                                          # Step 3'

        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
        
        return difference

    def vector_to_dictionary(self, theta, shapes):
        """
        Unroll parameters dictionary from a single vector satisfying our specific required shape.
        """
        parameters = {}
        previous_end = 0
        for key, shape in sorted(shapes.items(), key=lambda t: t[0]):
            parameters[key] = theta[previous_end: previous_end + shape[0]*shape[1]].reshape(shape)
            previous_end = previous_end + shape[0]*shape[1]
        return parameters

    def dictionary_to_vector(self, parameters):
        """
        Roll all parameters dictionary into a single vector.
        """
        shapes = {}
        count = 0
        for key,value in sorted(parameters.items(), key=lambda t: t[0]):
            
            # flatten parameter
            new_vector = np.reshape(value, (-1,1))
            shapes[key] = value.shape
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1

        return theta, shapes