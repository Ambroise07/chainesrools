"""
class Neuron  use the engine to update, 
the values of its weights.  

date : 5.10.2026
author: Gnabro Israel

"""
import engine
import random

    

class Neuron:

  def __init__(self, n_inputs:int):
    """
      initialise a neuron 

      Args:
        n_input: int --> number of input should be an integer
    """
    if not isinstance(n_inputs, int):
      raise TypeError('number of input should be int')

    self.ws : list = [engine.Scalar(random.uniform(-1, 1)) for _ in range(n_inputs)]    
    self.b : engine.Scalar = engine.Scalar(random.uniform(-1, 1))    

    # should be update when train the neuron
    self.x:list = []

    # target should have two items
    self.target:list = []
    

  def __call__(self, x:list):
    """ get the activation of the neuron """
    act =  sum((wi * xi for wi, xi in zip(self.ws, x)), self.b)
    self.X: list = x
    return act.tanh() 

  def parameters(self):
    """ return parameters """
    return self.ws + [self.b]

  def train(self, X, niter=5, lr=.01, callback=None):
    """ train the neuron """

    for i in range(niter):

      # zero grad
      for p in self.parameters():
        p.grad = .0
      
      # forward pass
      ypred = self(X)
      loss = (ypred - self.target) ** 2

      # backward pass
      loss.backward()
            
      # update parameters
      for i, p in enumerate(self.parameters()):
        p.data -= p.grad * lr
        d -= p.data

      # callback
      if callback:
        callback(loss)



class Layer:

  def __init__(self, n_neurons:int, n_inputs:int):
    """
      single layer with n_neurons of n_inputs.

      Args:
        n_neurons: int ---> number of neurons of the layer
        n_inputs: int ---> number of inputs of the neuron
        
    """
    if not isinstance(n_neurons, int):
      raise TypeError('number of input should be int')

    if not isinstance(n_inputs, int):
      raise TypeError('number of input should be int')  
                  
    self.neurons: set = { Neuron(n_inputs) for _ in range(n_neurons) }
    #self.n_inputs:int = n_inputs
    
  def __call__(self, x:list):
    out = [n(x) for n in self.neurons]
    return out
      
  def parameters(self):
    """ return all paramers of the neurons of layer """
    return [p for neuron in self.neurons for p in neuron.parameters() ]


  def train(self, niter:int):
    """ train the MLP """
    for neuron in self.neurons:
      neuron.train()


class MLP:
  def __init__(self, n_inputs, n_neurons:list):
    """
      a Multi layers Perceptrons.

      Args:
        n_inputs: int --->  number of inputs
        n_neurons: list ----> list of number of neurons in each layer
        
     """

    if not isinstance(n_inputs, int):
      raise TypeError('number of input should be int')
    
    if not isinstance(n_neurons, list):
      raise TypeError('neurons should be list[int]')

    size = [n_inputs] + n_neurons
    self.layers = { Layer(size[i+1], size[i]) for i in range(len(n_neurons)) }


  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x[0] if len(x) else x

     
  def parameters(self):
    """ get all parameters """
    return [p for layer in self.layers for p in layer.parameters()]

  def train(self):
    """ train the MLP """
    for layer in self.layers:
      layer.train()
