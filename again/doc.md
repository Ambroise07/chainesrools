# What is chainesrools?
**chainesrools** is a lightweight framework designed to build neural networks with ease. Its name is inspired by the **chain rule**, which is used to compute the gradients of neural network weights.

You do not need to install any external dependencies to use it. It features a minimalist API, making it quick and easy to understand.

---

# API 1: Scalar
The core building block of the framework is the `Scalar` object. It represents a single number (float) wrapped in a class that enables automatic differentiation. 

Each scalar tracks its lineage through a list called `children`, which stores the sub-scalars involved in the operation. The specific operation type is stored in the `op` attribute. This data helps propagate gradients and update other scalars when changes occur.

```python
>>> from chainesrools import Scalar
>>> a = Scalar(data=23.9)  # You can also write Scalar(23.9)
>>> print(a.children)
[]
```

The operation (`op`) is automatically set when creating new scalars through operations:

```python
>>> b = a + Scalar(23)
>>> print(b.op)
'+'
```

`Scalar` objects support all basic arithmetic operations. You can also perform operations combining a `Scalar` and a regular Python number (the returning value will automatically be wrapped into a `Scalar`).

```python
>>> c = b + 4
>>> print(c.data)
50.9
```

Each `Scalar` object holds a `grad` attribute used to accumulate gradients. By default, it is initialized to `0.0`.

```python
>>> a = Scalar(data=23.9)
>>> a.grad
0.0
```

You can update local gradients manually using the internal `_backward()` method. This only updates the immediate children's gradient values and does not propagate further back or affect the scalar itself.

```python
>>> # Set the seed gradient before calling _backward
>>> c.grad = 1.0 
>>> c._backward()
```

In the previous case, only the direct children of `c` are updated. To update the entire computational graph down to the very first inputs, you would normally need a nested loop:

```python
>>> # Manual deep update block
>>> for item in c.children:
...     for sub in item.children:
...         sub._backward()
...     item._backward()
```

While this works for simple expressions, it becomes impossible to manage manually for complex networks where the exact architecture is unknown. This automation is handled by the `Scalar.backward()` method. This method automatically builds a **topological graph** of all connected nodes and calls `_backward()` in the correct sequential order to calculate all gradients.

---

# API 2: Neuron
To start building networks, a built-in `Neuron` class is provided. All weights and biases inside a `Neuron` instance are stored as `Scalar` objects. 

The constructor expects the **number of inputs** (not the inputs themselves). To feed data into the neuron, call the instance as a callable object, passing the input vector as an argument:

```python
>>> from chainesrools import Neuron  
>>> from random import uniform
>>> X = [uniform(-1, 1) for _ in range(100)]
>>> n = Neuron(100)
>>> n(X)
# Returns a Scalar object representing the output activation
```

The `Neuron.train(X, target)` method expects an input vector `X` and a `target` value (which should be bounded between -1 and 1). For more details about input data requirements, please read the [Data Input](#data-input) section below.

To extract the internal parameters, call the `.parameters()` method, which returns a list of `Scalar` objects:

```python
>>> params = n.parameters()
```

You can achieve the exact same result manually by concatenating the weights list and the bias:
```python
>>> n = Neuron(12)
>>> all_params = n.ws + [n.b]  
```

---

# API 3: Layer
The `Layer` class represents a collection of multiple neurons running in parallel. 
* `nin`: The number of inputs feeding into a single neuron.
* `neu`: The total number of neurons present in this layer.

```python
>>> from chainesrools import Layer
>>> nin = 12
>>> X = [[uniform(-1, 1) for _ in range(nin)] for _ in range(100)]
>>> first_layer = Layer(nin, 42)
>>> first_layer(X)  
```

Calling `.train()` or executing the instance as a callable on a `Layer` object will automatically forward the operation and trigger the training mechanism across every individual neuron it contains.

---

# API 4: MLP (Multi-Layer Perceptron)
The `MLP` class connects multiple layers together. Just like the `Layer` class, it is fully callable and includes a `.train()` method.

The constructor expects:
1. An integer representing the number of input features for the **first layer**.
2. A list containing the number of neurons for each sequential layer \(k\).

```python
>>> from chainesrools import MLP
>>> nin = 12
>>> X = [[uniform(-1, 1) for _ in range(nin)] for _ in range(100)]
>>> NET = MLP(nin, [12, 43, 5, 2, 1])
>>> NET(X)  
```

---

# API 5: Use Cases & Demos
If you want to see a concrete application of this framework, you can run the built-in demos directly from your terminal:

```bash
python -m chainesrools test # You can also try vnet or lessnet
```

To see information and available configurations for a specific demo, add the `--help` flag after its name:

```bash
python -m chainesrools vnet --help 
```

---

# Data Input Requirements

All input features and targets **must be normalized within the range of [-1, 1]**. 

### Why is this necessary?
The underlying activation function used by default in this framework is the **hyperbolic tangent (tanh)**. 

The `tanh` function output strictly ranges from `-1` to `1`. When `loss.backward()` is applied, the model adjusts its weights to try to match your target. If your target is outside this boundary, the network will never converge.

For example, if your target is set to `43` and the maximum possible output of the model is `1` (due to `tanh`), the network will compute a high error, and gradients will keep updating indefinitely trying to solve this equation:
\[\text{loss} = 0 \iff (\tanh(n) - 43)^2 = 0 \iff \tanh(n) = 43\]

This mathematical statement is impossible to satisfy because `tanh` stops at `1`. Therefore, all data fed into the neurons must strictly range from `-1` to `1`.

*Note:* The framework also supports the `sigmoid` activation function. If you need a custom activation function, feel free to implement it and contribute!

To visualize any network architecture you have built, you can call the `.show()` method on any `Layer`, `Neuron`, or `MLP` object. See the [NNplots](#nnplots) section for more details.

---

# NNplots
The `NNplots` module provides visual representations of your created neural networks. It uses **Graphviz** under the hood to generate graphs.
