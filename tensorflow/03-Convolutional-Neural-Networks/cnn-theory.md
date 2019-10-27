# Convolutional Neural Networks

## Theory

- Based on biological research of visual neurons
- Neurons in the visual cortex have a small local receptive field

## Tensors

- N-Dimensional arrays
- Data types:
  - Scalar: integer - 3
  - Vector: 1-d array - [3,4,5]
  - Matrix: 2-d array:
  - [[3,4], [5,6], [7,8]]
  - Tensor: higher-dimensional arrays:
  - [[[3,4], [4,5]], [[6,7], [8,9]]
- Tensor in mnist:
  - I - images
  - H - Height
  - W - Width
  - C - Color
- Instead of a deep neural network, where every neuron in layer n is connected to every neuron in layer n+1...
- A convolutional neural network is only connected to a few nearby neurons in the next layer
- This is helpful for dealing with large inputs, such as images (56k pixels would take forever with a deep neural network)
- Also helpful because nearby pixels in images are much more related than distant pixels
- Deep learning
  - Each CNN layer looks at increasingly large parts of the image
  - Having only those close connections aids in invariance
  - Help with regularization, limits the weights
- 1-d convolution:

$$
y = w_1+x_1 + w_2+x_2
$$
if
$$
(w_1, w_2) = (1, -1)
$$
then
$$
y = x_1 - x_2
$$
and y is at a maximum in
$$
(x_1,x_2) = (1, 0)
$$
This is how we will handle padding the images with 0s

## Pooling Layers

Remove most of the data from the input in order to make it computationally feasable

## Dropout

- Regularization to prevent overfitting
- Randomly removes neurons to prevent coadaptation to a particular data set
