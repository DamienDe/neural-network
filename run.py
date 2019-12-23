import component.layers as ly
import component.network as nw
import test as test

X, Y, test_x, test_y, classes = test.load_work()
#Parameters
lambd = 0.2
coef_beta = 0.9
learning_rate=0.0075
nb_iterations=2000
keep_prob = 0.6
gradient_check=True

#Layers
# hidden_layers = [ly.NormalizedReluLayerRegularized(test_x.shape[0], 3, lambd=lambd, coef_beta=coef_beta), ly.NormalizedReluLayerRegularized(3,2, lambd=lambd, coef_beta=coef_beta), ly.NormalizedSigmoidLayer(2, 1,coef_beta=coef_beta)]
hidden_layers = [ly.NormalizedReluLayerRegularized(X.shape[0], 3, lambd=lambd, coef_beta=coef_beta), ly.SimpleReluLayerWithDropOut(3,2, keep_prob=keep_prob), ly.NormalizedSigmoidLayer(2, 1,coef_beta=coef_beta)]


#Network
nn = nw.Network(hidden_layers)

#Training
parameters = nn.train(X=X, Y=Y,learning_rate=learning_rate, nb_iterations=nb_iterations, gradient_check=gradient_check)

#Prediction on train
nn.predict(X=X, Y=Y, prediction=True, parameters=parameters)

#Prediction on test
nn.predict(X=test_x, Y=test_y, prediction=True, parameters=parameters)
