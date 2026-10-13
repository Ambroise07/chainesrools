from  neurons import Neuron, MLP
from random import uniform, randint
from pprint import pp as print
import math

def print_(e):
  pass

"""
X = [uniform(-1, 1) for i in range(100)]
n = Neuron(100)
n.train(target=-1, x=X,  callback=print_)

print(f"output: {n(X)}  ---  target is -1")
"""

def sin():
  """ guess sinus values with a neuron """
  ninput = 10000
  Y = [math.sin(i) for i in range(ninput)]
  X = [i*0.1 for i in range(ninput)]
  mlp = MLP(ninput, [4, 5, 1])
  mlp.train(target=Y)

  
  print(f"output: {mlp(X)}  ---  target is -1")

sin()
