import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Operation():
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.

    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """

    def __init__(self, input_nodes=[]):
        """
        Intialize an Operation
        """
        self.input_nodes = input_nodes  # The list of input nodes
        self.output_nodes = []  # List of nodes consuming this node's output

        # For every node in the input, we append this operation (self) to the list of
        # the consumers of the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)

        # There will be a global default graph (TensorFlow works this way)
        # We will then append this particular operation
        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        """ 
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        """
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_mat, y_mat):
        self.inputs = [x_mat, y_mat]
        return x_mat.dot(y_mat)


class Placeholder():
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """

    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph():  # This is the global graph that keeps track of everything
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order (Ax first , then Ax + b).
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:
    def run(self, operation, feed_dict={}):
        """ 
        operation: The operation to compute
        feed_dict: Dictionary mapping placeholders to input values (the data)  
        """
        # POut nodes in correct order
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:  # Operations are here
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

        # Covert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


#  Add up some numbers and multiply!
g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)

x = Placeholder()
y = multiply(A, x)

z = add(y, b)

sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})
print(result)

# Matrix multiplication!
g = Graph()
g.set_as_default()

A = Variable([[10, 20], [30, 40]])
b = Variable([1, 1])

x = Placeholder()

y = matmul(A, x)

z = add(y, b)

sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})
print(result)

# Activation functions


def sigmoid(z):
    return 1/(1+np.exp(-z))


# This is just plotting the sigmoid function to show whats up
sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)
plt.plot(sample_z, sample_a)
plt.show()


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1/(1+np.exp(-z_val))


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
features = data[0]
labels = data[1]

print(np.array([1, 1]).dot(np.array([[8], [10]])) - 5)
print(np.array([1, 1]).dot(np.array([[4], [-10]])))

# Draw a line that separates the classes
x = np.linspace(0, 11, 10)
y = -x + 5
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)
plt.show()

g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1, 1])  # Matrix [1, 1] to multiply our feature matrix agzinst
b = Variable(-5)
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
result = sess.run(operation=a, feed_dict={x: [8, 10]})
print(result)
nextResult = sess.run(operation=a, feed_dict={x:[0, -10]})
print(nextResult)
