"""
 neurons module. 
 use this to test the engine on tasks.
 
 -------------------------------------
 date : 7.10.2026
 author: Gnabro Israel
 
"""

try:
    # Cas 1 : Utilisé lorsque le projet est lancé depuis la racine (ex: python cases.py)
    from core.engine import Scalar
except ModuleNotFoundError:
    # Cas 2 : Utilisé si vous lancez le fichier neurons.py directement tout seul
    from engine import Scalar


from random import uniform
from collections.abc import Callable 
# should import graphizx here
 
"""
like pythorch zeros_grad method resert to 0.0 the grad 
this is intended. 

"""


class Module:

  def zero_grads(self):
    """ resert the grad to zero """

    for p in self.parameters():
      p.grad = 0.0


class Neuron(Module):

  """ 
    Neuron hold just weights and bias whichs are Scalar objects.  
    the numbers of inputs is called 'nin', crazy like name :).
    
  """

  def __init__(self, nin:int):
    if not isinstance(nin, int):
      raise TypeError(f'number of input should be int not {type(nin)}')     
    self.nin = nin
    self.ws = [Scalar(uniform(-1, 1))  for _ in range(nin)]
    self.b = Scalar(uniform(-1, 1))
    self.lr = .001 


  def parameters(self):
    """ get parameters of neuron """
    return self.ws + [self.b]

  
  def __call__(self, x:list):
    """
      I like call n(x), because is more intuitive 
      than n.activate(x)

      So this will compute the activation of the neuron
      I use tanh, but you can update the code and use sigmoid for 
      instance.   
    """
    out = sum( (wi * xi for wi, xi in zip(self.ws, x)), self.b )
    return out.tanh()


  def train(self, x:list, target:int|float, 
            lr:float=.001, niter:int=2000, callback:Callable=None):
    """

      You can directly train a single neuron, without write 
      any training loop.

      But if you want to write one, don't use this method.
      target argument should be between -1 and 1.
       
    """
    # get data 
    if not isinstance(target, (float, int)):
      raise TypeError(f"""
      Target is float or int, not {type(target)} please read documentation
      for details.
      """)


    if not isinstance(x, list):
      raise TypeError(f'{x} should be a list.')

    if not isinstance(lr, float):
      raise TypeError(' bad learning rate: should be float  ')

    
    if not isinstance(niter, int):
      raise TypeError(' number of iteration should be int  ')  


    if not niter>0:
      raise ValueError(' number of iteration should be granther than 0. ')

    
    if not callable(callback):
      raise TypeError(' callback should be callable.') 

    # get the learnin rate
    self.lr = lr 

    # run the train loop    
    for _iter in range(niter):

      # resert the grad to zero
      self.zero_grads()

      # forward pass
      ypred = self(x)
      loss = (target - ypred)**2

      #backward pass
      loss.backward()
      for p in self.parameters():
        p.data -= p.grad * self.lr

      callback(loss)

  def show(self):
    """ 
    display the Neuron with its weights and input data using
    graphviz

    """
    pass 


class Layer:

  def __init__(self, nin:int, neu:int):
    """
      the layer is a stack of neurons not connected.

      Args:
        nin: number of inputs of a single neuron.
        neu: number of neurons.
        
    """
    if not (isinstance(nin, int) and isinstance(neu, int)):
      raise TypeError('number of neurons and inputs are both integers.')

    self.neurons = [Neuron(nin) for _ in range(neu)]


  def __call__(self, x:list):
    """
    Get the activation of each neuron of 
    this layer.
    
    """
    if not isinstance(x, list):
      raise TypeError(" x should be list. Read  docs for more details.")

    """
      x can be a  nested list. In this case, the if stament, 
      is provided to ensure that the Layer can perform this 
      data.

      e.i : 
        x = [ [], [], [],...[]] (supported)
        x = [ [ [ [ []]]], [[]], [],...[]] (unsupported)
      
    """
    if all(isinstance(xi, list) for xi in x):
      return [ n(xi) for xi in x for n in self.neurons ]            

    return [n(x) for n in self.neurons]    

    
  def train(self, **params):
    """ 
    train a single Layer 

    params hold arguments of the training 
    method of each neuron (it's called train).
    read neuron.train signaturs.
    
    """
    for n in self.neurons:
      n.train(**params)
      

  def show(self):
      """ 
      display the Neuron with its weights and input data using
      graphviz
  
      """
      pass


  def parameters(self):
    """ return parameters """
    return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
  """
  nin : number of input (for each neuron in a layer).
  nlayers: list of layers.
  
  """
  def __init__(self, nin:int, nlayers:list):
    size = [nin] + nlayers
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(nlayers))]
    self.__nin = nin


  def __call__(self, x: list):
  
    if not (isinstance(x, list)):
      raise TypeError("x should be a list.")

    if not all([isinstance(xi, list) and len(xi) == self.__nin for xi in x]):
      raise ValueError("""
      Expected inputs should be contains lists as item.
      read docs for more details about inputs. 
      """)
       
    out = x
    for layer in self.layers:
        out = layer(out)
        
    return out




  # Dans core/neurons.py (Tout en bas, dans la classe MLP)
  def train(self, x: list, target: list, callback=None, lr: float = .001, niter: int = 1):
    """
    Trains the entire MLP by iterating through each sample 
    and passing individual examples down to the layers.
    """
    if not isinstance(x, list) or not isinstance(target, list):
        raise TypeError("x and target must both be lists containing your dataset samples.")
        
    # On boucle sur le nombre d'itérations globales (epochs) de l'entraînement
    for _epoch in range(niter):
        # On fait défiler les images une par une avec leur cible associée
        for xi, yi in zip(x, target):
            # On transmet l'échantillon individuel aux couches
            for l in self.layers:
                l.train(x=xi, target=yi, callback=callback, lr=lr, niter=1)



  def show(self):
      """ 
      display the Neuron with its weights and input data using
      graphviz
  
      """
      pass

  def parameters(self):
    """ return parameters """
    return [n for layer in self.layers for n in layer.parameters()]

    
