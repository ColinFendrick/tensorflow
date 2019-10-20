# Neural network notes

Perceptron:
Takes in inputs, gives each input weight, and handles output of sum through activation function

Input A (12) --- (weight 0.5)
Input B (6) --- weight -1 
=== f(4) => [does something in activation function] => res of f(4)

Can give bias also(ie add 1 to value)

$$
  \sum_{i=0}^n  w_ix_i + b
$$
where n = inputs, w=weight, x=value, b=bias

## Multiple Perceptrons Network

### Input layer, hidden layers, then output layer

- Input layer = real values from data
- Hidden layers = manipulate data
- Output layer = final output

### Activation function

eg:

Sigmoid function
$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Hyperbolic tanget, hyperbolic sine, cosine

$$
tanh(x) = \frac{sinh(x)}{cosh(x)}
$$

$$
cosh(x) = \frac{e^x + e^{-x}}{2}
$$

$$
sinh(x) = \frac{e^x - e^{-x}}{2}
$$

Rectified Linear Unit (ReLU):

$$
ReLU(x) = max(0, z)
$$
