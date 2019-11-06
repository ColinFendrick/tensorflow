# RNN

Often uses for sequences:

- Data over time
- Language
- Position of cars/traffic
- Speech/audio
  
## Whereas normal neurons take in aggregation of inputs, uses an activation fn, and sends output, a recurrent neuron sends the data back to itself

Cells that are functions of inputs from previous timesteps are known as memory cells

RNN are flexible in inputs and outputs

## Vanishing Gradient

Backpropogation goes backwards from output to input layer, propogating error gradient.

For deeper networks issues can arise from backpropogation: vanishing/exploding gradients.

### Why does this happen

For eg sigmoid:

$$
f(x) = \dfrac{1}{1+ e^{-x}}
$$

Large inputs give gradients very close to 0. So the gradient will decrease exponentially as you increase layers while the front layers train very slowly.

### Solving

- Different activation functions :Leaky ReLU, or Exponential Linear Unit, etc
- Batch normalization where model will normalize each batch using batch mean and std dev
- Gradient clipping can also work (fix gradient for [-1, 1])
  
### In RNN

The above solutions sometimes don't work in RNN because of the length of time series inputs, the training will take increasingly long

We can shorten time steps, but this makes the model worse at predicting longer trends

Another issue RNNs face is the network "forgets" the first inputs, as information is lost at each step through the RNN

We need long term memory

This is how the Long Short-Term Memory Cell works

### LSTMC

These cells have a cell state that gets returned and fed back in

So the cell takes in

$$
x_t,h_{t-1}, c_{t-1}
$$

and outputs

$$
c_t, h_t
$$

The first step is to feed input and previous output through our forget-gate layer:

$$
f_t = \sigma (W_f\cdot[h_{t-1},x_t] + b_f)
$$

Takes in the previous output and current input, with weights and biases, and puts it through a sigmoid function

Now we decide what to store in the cell state:

Input-gate layer is a sigmoid layer:

$$
i_t = \sigma(W_i\cdot[h_{t-1},x_t] + b_i)
$$

Then we take a hyperbolic tangent layer to produce a vector of possible values:
$$
\tilde{C_t} = \tanh(W_C\cdot[h_{t-1},x_t] + b_i)
$$

Combine these two to update the cell state

To update old cell state to new cell state:

$$
C_t = f_t\ast C_{t-1} + i_t\ast \tilde{C_t}
$$

That is: forget-gate times old state plus input-gate layer times candidate values

Then output is:

$$
o_t = \sigma(W_o\cdot[h_{t-1},x_t] + b_o)
$$
$$
h_t = o_t\ast (C_t)
$$

Can also allow peepholes, which passes in previous cell states

Or the gated recurrent, which combines forget/input gates into one
