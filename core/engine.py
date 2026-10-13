# understabd the chain rul apply in engine.py - ok
# read the chaine rul article on wiki 

from __future__ import annotations

"""
The engine code src. 
you can install it 'pip install ...'
date : 6.10.2026
author: Gnabro Israel.
 
"""
import math


class Scalar:

  def __init__(self, data:float=.0,  label:str='', op:str='', children:list=[]):

    """
      Scalar is similar to Value like in micrograd repo or 
      torch.tensor.

      It come with all required operations to update the value
      that it hold.

      *label* help recongize this scalar when using framework like 
      graphviz to display the node.

      when do the backward pass, the operators  should be 
      updated, so the children is provided to get these previous
      Scalar objects.

      Every _backward method follow the chain rule to update
      scalars. [link to chain ruls]
       
    """

    # get the data
    if not isinstance(data, (int, float)):
      raise ValueError(f"data should be a int or float not {type(data)}")
    self.data = data

    # get the label
    if not isinstance(label, str):
      raise TypeError(f"label should be  str not {type(label)}")
    self.label = label

    # get the op
    if op not in ['*', '**', '+', '-', '', 'tanh', 'sigmoid']:
      raise ValueError(f"op should be in ['*', '**', '+', '-', 'tanh', 'sigmoid']")
    self.op = op

    # get the children
    if not isinstance(children, list):
      raise TypeError(f"children should be a list. Not {type(children)}")
    self.children = set(children)

    # nodes 
    self.nodes = []
    self.grad = .0
    self._backward = lambda : None


  def __convert(self, data:int|float):
    """ convert data into a scalar object """
    if not isinstance(data, Scalar):
      return Scalar(data=data, label=str(data))
    return data
 
  def __repr__(self):
    return f"Scalar({self.data})" 
  
  def __add__(self, other:int|float|Scalar):
    """ add two scalars together """
    other = self.__convert(other)
    _sum = self.data + other.data
    _label =f'{self.label}+{other.label}'

    out = Scalar(data=_sum, label=_label, op='+', children=[self, other])

    """
    As you can see, call _backward of out, will increase the 
    data attributes of the operators by an amount (out.grad)

    do not call this method, use instead backward one.
    """
    

    def _backward():
      """ Update children data """
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    
    out._backward = _backward
    return out

      
  def __mul__(self, other:int|float|Scalar):
    """ product of two scalars. """
    other = self.__convert(other)
    _prod = self.data * other.data
    _label =f'{self.label}x{other.label}'

    out = Scalar(data=_prod, label=_label, op='*', children=[self, other])

    """
    As you can see, call _backward of out, will increase the 
    data attributes of the operators by an amount (out.grad)

    do not call this method, use instead backward one.
    """

    def _backward():
      """ Update children data """
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    
    out._backward = _backward
    return out


  def __sub__(self, other:int|float|Scalar):
    """ substraction of two scalars """
    other = self.__convert(other)
    _sub = self.data - other.data
    _label =f'{self.label}-{other.label}'

    out = Scalar(data=_sub, label=_label, op='-',  children=[self, other])

    """
    As you can see, call _backward of out, will increase the 
    data attributes of the operators by an amount (out.grad)

    do not call this method, use instead backward one.

    Of course, the __neg__ can be implemented, but for simplicity reason 
    i didn't do it.
    """

    def _backward():
      """ Update children data """
      self.grad += 1.0 * out.grad
      other.grad += -1.0 * out.grad
    
    out._backward = _backward
    return out


  def __pow__(self, n:float|int):
    """ this scalar to the power of n """
    _pow = self.data ** n
    _label =f'{self.label}**{n}'

    out = Scalar(data=_pow, label=_label, op='**',  children=[self])

    """
    As you can see, call _backward of out, will increase the 
    data attributes of the operators by an amount (out.grad)

    do not call this method, use instead backward one.

    Of course, the __neg__ can be implemented, but for simplicity reason 
    i didn't do it.
    """

    def _backward():
      """ Update children data """
      self.grad += ( n * (self.data**(n-1)) )  * out.grad
    
    out._backward = _backward
    return out


  def __truediv__(self, other:int|float|Scalar):
    """ 
    divide self by other can be convert into a product:
    self * (other ** (-1)) ----> self / other 
    """
    other = self.__convert(other)
    _div = self * (other ** (-1) )
    _label = f'{self.label}/{other.label}'

    out = Scalar(data=_div, label=_label, op='/', children=[self, other])

    """
    As you can see, call _backward of out, will increase the 
    data attributes of the operators by an amount (out.grad)

    do not call this method, use instead backward one.

    Of course, the __neg__ can be implemented, but for simplicity reason 
    i didn't do it.
    """

    def _backward():
      """ Update children data """
      self.grad += ( n * (self.data**(n-1)) )  * out.grad
    
    out._backward = _backward
    return out                 

  """ 
  the next methods are used by neuron to compute it activation 
  This part will be update soon.
  """

  def tanh(self):
    """ compute the tanh """
    _label = f'tanh({self.label})'
    _tanh = math.tanh(self.data)
    
    t = Scalar(data=_tanh, label=_label, op='tanh', children=[self])

    def _backward():
      self.grad += (1 - t.data**2)  * t.grad
    t._backward = _backward

    return t   

  def sigmoid(self):
    """ compute the sigmoid func """
    _data = (1 / 1 + math.exp(self.data))
    _label = f"sigmoid({self.label})"
    sig = Scalar(data=_data, label=_label, op='sigmoid', children=[self])

    def _backward():
      self.grad += sig.data * (1 - sig.data)  * t.grad
    t._backward = _backward
    
    return t

#------------------------------------------------------------------------------#
# backward method : use this instead of _backward... 
#------------------------------------------------------------------------------#
  def backward(self):
    """ update the gradients of the children of this node """
    visited = set()

    """
    this function is call build_topo in other lib,
    but here it call build_nodes.
    
    """
    def build_nodes(parent): 
          
      if parent not in visited:
        visited.add(parent)
        for child in parent.children:
          build_nodes(child)
        self.nodes.append(parent)

    self.grad = 1.0
    build_nodes(self)

    for scalar in reversed(self.nodes):
      scalar._backward()
      
            
           


#------------------------------------------------------------------------------#
# previous methods are equivalent of these
#------------------------------------------------------------------------------#    

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    other = self.__convert(other)

    """
     because other is a scalar too,
     other.__sub__ is called when doing
     the follow operation:
    """
    return other + (self * -1)


if __name__=='__main__':

  """
   if this file is executed, the tests will be display 
   as output.
  
   Tests are wrote in test.txt file. 
  """
  #import doctest
    
  #doctest.testfile('test.txt')

  number = Scalar(124.888) + 3
  number.grad = 1.0
  number._backward()
  print([it.grad for it in number.children])
