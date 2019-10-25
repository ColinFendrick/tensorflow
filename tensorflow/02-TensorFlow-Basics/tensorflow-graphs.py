import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)

n3 = n1 + n2
with tf.Session() as sess:
    result = sess.run(n3)
    print(n3)

graph_one = tf.get_default_graph()
graph_two = tf.Graph()

with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

print(graph_two is tf.get_default_graph())
