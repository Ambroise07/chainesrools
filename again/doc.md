# what's chainesrools
Chainesrools is a little framework to create NN with ease. It name is 
inspired by the _chaines rule_, which is used to compute the gradiens 
of NN weights.

You don't need to install any things else to use it. It has a few apis, so 
understand each of thems will be quick.


# API 1 : Scalar
The first thing you should know, is _Scalar objects_, which represente 
numbers (float), wrapping into class that enable it to be update by others 
scalars. Each scalar has  a list named 'children', this list stores 
operators. The operand is stored in 'op' attribute, these datas, help to 
update others scalars when this changed. 

"""
 >>> a = Scalar(data=23.9)  # you can also write Scalar(23.9) 
 >>> print(a.children)
 ''
"""

the operand is also set into the scalar:

"""
 >>> b = a + Scalar(23) 
 >>> print(b.op)
 '+'
"""

Scalar object support all arithmetic operations, and someone can add a scalar
and a number together (the returning value, will be a scalar).

"""
 >>> c = b + 4  # you can also write Scalar(23.9) 
 >>> print(c.data)
 ''
"""

Scalar object hold a grad attribute that's use to update the value of the data
by defaiult it set to 0.0

"""
 >>> a = Scalar(data=23.9)  # you can also write Scalar(23.9) 
 >>> a.grad
 0.0
 
"""

We can update scalars with the _backward method. this will only update the 
children data values and will not affect the scalar itself.

"""
 >>> values = [item.data for item in c.children]
 >>> "(before updating) the value of data ", values

 >>> c.grad = 1.0 # by default the value is 0.0, you should set it to 1.0
 >>>              # before calling _backward method.
 
 >>> c._backward()  # you can also write Scalar(23.9) 
 >>> values = [item.data for item in c.children]
 >>> "(after updating) the value of data ", values
 
"""
In the previous case, only b and another scalar is update. Can we update to
the children of b and the last scalar ? You may use a loop to achive that:  

"""

 >>> # bloc
 >>> for item in c.children:
 ...  for sub in item.children:
 ...    sub._backward()
 ...  sub._backward()

""" 


Of course, this work for the scalar c, but how to call the _backward method
when we aren't define the scalar or don't know how is defined ? We should
also call the '# bloc' on the subitem call 'sub'. That's task is handled by
_Scalar.backward_ method. The method will build a topological graph of the 
children of this scalar and then call the _backward method to update the 
float value.


# API 2: Neuron

To start experimentation, a Neuron class is provided. so you don't need to 
create a new class for that.
All weights of the Neuron instance are scalars objects. The class come with 
it train method, the constructor expected the number of input 
(not inputs itselves). To feed it input, call the instance as a callable
object where the argument is input:


"""

 >>> from chainesrools import Neuron  
 >>> from random import uniform
 >>> X = [uniform(-1, 1) for _ in range(100)]
 >>> n = Neuron(100)
 >>> n(X)
 ''
 >>> n.train(X, target=-1)

 
"""

The chainesrools.Neuron.train method, expected unless the X value and the
target number (which is in the bounds of -1 and 1). For more details, about
inputs datas, please read here.
To get the parameters, call the  .parameters() method, which return a list 
of Scalars [link to scalar]. 
You can also get the same result by doing:
 
 """
 >>> n = Neuron(12)
 >>> n.ws + [n.b]  

 """

To more informations about this chainesrools.Neuron please read here


# API 3: Layer
The Layer, creates multiple neurons. the number of single neuron input is 
give into the __init__ method and stores into nin argumen, the number of 
expected neurons is fed into neu argument of the constructor.


 """
 >>> nin = 12
 >>> X = [ [uniform(-1, 1) for _ in range(nin)] for _ in range(100)]
 >>> first_layer = Layer(nin, 42)
 >>> first_layer(X)  

 """

 You can also call .train() on a Layer object. As you call .train, this object
 will can the .train method of each neuron. Same when you call Layer instance
 as a callable.
 
 
There no more informations about this class, but take a look there (link).

# API 4: MLP

MLP come with many layers connected. As Layer class, is has a train method.
It's a callable. The constructor expects a interger: the numbers of inputs
data of the _first layer_ and a list, which's stores the number of neurons 
of layer k (k is the index of the number in the list).

 """
 >>> nin = 12
 >>> X = [ [uniform(-1, 1) for _ in range(nin)] for _ in range(100)]
 >>> NET = MLP(nin, [12, 43, 5, 2, 1])
 >>> NET(X)  

 """

fore more details please read here...



# API 5: usecase

If you want see the real application of this framework, please have a look:

"""
 >>> python -m chainesrools test # you can also try vnet or lessnet

"""

if you want see information about each demo, add the help flag after the demo
name

"""
 >>> python -m chainesrools vnet --help 

"""


# DATA INPUT:

 All data should be in range 1 (from -1 to 1). It's same thing with the 
 target value passed in the .train method. Why ? Because the underline 
 activation function use is tanh.

 tanh start at -1 and end 1. So when the loss.backward is applied, to reach
 the target value, the modal will not find the right weight for the problem
 if target is not convert into a value from -1 to 1.

 Of course, assume that the target is 43 and the output of the modal is 
 0.5428899 (which is bounce in -1 and 1), the loss.backward() method is 
 called and the error is different to 0, so the weights will be update too
 to give the tanh function (used to compute the activation of modal) a 
 number n,  so that :
   loss = 0  eq (tanh(n) - target)**2 = 0
             eq tanh(n) = target = 43

 the last statement is impossible to achive.  Because tanh, can reach the 
 number 43 (It stop at 1! ) Any data fed into neurons should be from -1 to 
 1.

 there's another function  you can use to compute the activation, the 
 __sigmoid__ . If you want another function, feel free to add it.

 To see a network, that you created, call the .show method on object.
 Object should be Layer, Neuron or MLP. Read NNplots for more informations.   


# NNplots

NNplots display the neurons network that you created. It's use graphizx 
under the wood.   
