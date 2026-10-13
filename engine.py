"""
compute the gradiens of the neurones...

author : Gnabro Israel
date : 2.10.2026
 
"""
import math


class Scalar:
  """
  op should be +, -, *, ** and tanh.
  data should be an interger or float
  
  """

  def __init__(self, data:float, label:str='',  children:list=[],  op:str=''):

    if not isinstance(data, (int, float)):
      raise TypeError(f'expected integer or float value, but got {type(data)}')

    if not isinstance(op, str):
      raise TypeError(f'expected string, but got {type(data)}')

    if not isinstance(children, list):
      raise TypeError(f'expected list, but got {type(children)}')

    if not isinstance(label, str) and not label.strip() :
      raise TypeError(f'expected str, but got {type(children)} label should not be empty')          
    
    self.data = float(data)
    self.op = op
    self.grad = 0.0
    self._backward = lambda : None
    self.children = set(children)
    self.label = label
    self.nodes = []

    
  def __repr__(self):
    return f"Scalar(data={self.data})"    

  def __convert(self, data:int|float):
    """ convert data into this type """
    if not isinstance(data, Scalar):
      return Scalar(data, label=str(data))   
    return data

  def __mul__(self, other):
    other = self.__convert(other)
    result = Scalar(self.data * other.data, op='*', children=[self, other])
    result.label = f'{other.label}x{self.label}'
    
    def _backward():
      self.grad += other.data * result.grad
      other.grad += self.data * result.grad
    result.grad += 1.0
    result._backward = _backward
        
    return result   
      
  def __rmul__(self, other):
    return self.data * other      

  def __add__(self, other):
    other = self.__convert(other)
    result = Scalar(self.data + other.data, op='+', children=[self, other])
    result.label = f'{other.label}+{self.label}'
    
    def _backward():
      self.grad += 1.0 * result.grad
      other.grad += 1.0 * result.grad
    result._backward = _backward
    return result   
  
  def __radd__(self, other):
    return self + self.__convert(other) 

  def __sub__(self, other):
    other = self.__convert(other)
    result = Scalar(self.data - other.data, op='-', children=[self, other]) 
    result.label = f'{self.label}-{other.label}'

    def _backward():
      self.grad += 1.0 * result.grad
      other.grad += -1.0 * result.grad
    result._backward = _backward

    return result 

  def __rsub__(self, other):
    result = Scalar(self._convert(other) - self.data, op='-', children=[self, other]) 
    result.label = f'{other.label}-{self.label}'

    def _backward():
      self.grad += -1.0 * result.grad
      other.grad += 1.0 * result.grad
    result._backward = _backward
    
    return result 


  def __truediv__(self, other):
    other = self.__convert(other)
    return Scalar(data=(self.data * other**-1), 
                  op='div', label=f'{self.label}/{other.label}', 
                  children=[self, other])

  def __pow__(self, n):

    if not isinstance(n, (float, int)):
      raise TypeError('types of the power should be int or float. ')

    result = Scalar(self.data ** n, op='**', children=[self])
    result.label = f'{self.label}**n'
    
    def _backward():
      self.grad += (n * self.data**(n-1)) * result.grad
    result._backward = _backward
    return result

  def tanh(self):
    t = Scalar(math.tanh(self.data), op='tanh', children=[self])
    
    def _backward():
      self.grad += (1 - t.data**2) * t.grad

    t._backward = _backward
    return t
    
  def backward(self):
    """ update the gradiens of self and its children """
    visited = set()

    def build_topo(parent):
      # add only on time the parent 
      # power of set :)
      if parent not in visited:
        visited.add(parent)
        for child in parent.children:
            build_topo(child)
        self.nodes.append(parent)

    
    build_topo(self)

    self.grad = 1.0    
    for node in reversed(self.nodes):
      node._backward()
      print('node', node)    
    
  def print(self, data='nodes'):
    """ print the nodes """
    print('[')
    match data:
      case 'nodes':
        for node in self.nodes:
          print(' ', node)
      case 'grads':
        for node in self.nodes:
          print(' {0}.grad = {1} '.format(node.label, node.grad) )
      case _:
        raise TypeError("data should be in ['nodes', 'grads'] ")
    print(']')
