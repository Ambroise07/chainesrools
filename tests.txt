"""
test the engine code to not run this file.
use : python -m chainesrolls instead.

date : 6.10.2026
"""

# import the lib
 >>> from engine import Scalar

#first create a Scalar object
 ...
 >>> a = Scalar(23.0)
 >>> a
 Scalar(23.0)


#adding a scalar with a number will return a scalar 
 ...
 >>> b  = a + 12
 >>> b
 Scalar(35.0)


# call the _backward method will update the grad of the scalar
# children.
 ...
 >>> b.grad = 1.0
 >>> grads = [ item.grad for item in b.children ]
 >>> grads 
 [0.0, 0.0]

 ...
 >>> b._backward()
 >>> grads = [ item.grad for item in b.children ]
 >>> grads 
 [1.0, 1.0]


# so we can tests others operations
 >>> "a x b is ",  a * b 
 ('a x b is ', Scalar(805.0))

 >>> print("a - b is ",  a - b) 
 a - b is  Scalar(-12.0)

# test of MLP
>>> import random
>>> import pprint
>>> x = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(6)]
>>> from neurons import MLP 
>>> mlp = MLP(10, [12, 4, 10, 1])
>>> print(mlp(x))
