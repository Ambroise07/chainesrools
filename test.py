#from engine import Scalar
from pprint import pp
from neurons import Neuron, Layer, MLP 
import random

random.seed(42)

def _print(datas):
    """ print the datas """

    if not isinstance(datas, list):
      print(datas)
      return 
      
    print('[')
    for data in datas:
      print(' ', data)
    print(']')



"""
# with scalar side 
add = Scalar(12.3) + 5

print('addition', add)

sub = Scalar(12.3) - 5

print('substraction', sub)


mul = Scalar(12.3) * 5

print('multiplication', mul)



# with number side 
add = 5 + Scalar(12.3)

print('addition', add)

sub = 5 - Scalar(12.3)

print('substraction', sub)


mul = 5 * Scalar(12.3)

print('multiplication', mul)
"""

"""
b = 24 * Scalar(100)
print('the op of b is ', b.op)
"""

"""
b = Scalar(1) ** 1
print('the op of b is ', b.op)
print('the tanh of b is ', b.tanh())
"""

"""
# let see one sample

scalars = [Scalar(s) for s in range(10)]
results = []

for i in range(8):
  t = scalars[i] * scalars[i+1]
  t._backward()
  results.append(t)
  pp( f'result: {t}, grads : ( {scalars[i].grad}, {scalars[i+1].grad} )' )
  
  

  t = scalars[i] + scalars[i+1]
  t._backward()
  results.append(t)
  pp( f'result: {t}, grads : ({scalars[i].grad}, {scalars[i+1].grad} )' )
  

  t = scalars[i] - scalars[i+1]
  t._backward()
  results.append(t)
  pp( f'result: {t}, grads : ({scalars[i].grad}, {scalars[i+1].grad} )' )
  
  
  t = scalars[i].tanh()
  t._backward()
  results.append(t)
  pp( f'result: {t}, grads : ({scalars[i].grad}, {scalars[i+1].grad} )' )



print('total scalars in result', len(results))
pp(results, indent=4)  

"""

"""
a, b = Scalar(23.0, label='a'), Scalar(22.3, label='b')

c = a * 24 

d = c * b 

print('Expression : ', c.label)
c.backward()
c.print('nodes')
c.print('grads')

print(' \n \n')
print('Expression : ', d.label)
d.backward()
d.print('nodes')
d.print('grads')
"""


"""
print('two scalrs:', a, b)
print('action: c = a * b')
print('the gradien of a and b before  backward call', a.grad, b.grad)

# update the gradiens of d and b.
c._backward()

print('the gradien of a and b after backward call', a.grad, b.grad)

d = c * b * a
print('action: d = c * b * a')
print('the gradien of a , b and c  before backward call')
print(f"a[{a.grad}], b[{b.grad}] and c[{c.grad}]")

d._backward()
print('the gradien of a , b and c  after backward call')
print(f"a[{a.grad}], b[{b.grad}] and c[{c.grad}]")
"""
n = 2
x = [ random.uniform(-1, 1) for _ in range(4) ]
n = Neuron(n)
n.target = .5
n.train(callback=_print, X=x)

print('the output of neurone', n(x))
print('the target', n(x))

