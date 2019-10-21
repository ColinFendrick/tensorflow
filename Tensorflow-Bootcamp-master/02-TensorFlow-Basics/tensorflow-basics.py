import tensorflow as tf

hello = tf.constant('Hello')
world = tf.constant('World')
result = hello + world
with tf.Session() as sess:
    result = sess.run(hello + world)
    print(result)

a = tf.constant(10)
b = tf.constant(20)
with tf.Session() as sess:
    result = sess.run(a + b)
    print(result)

const = tf.constant(10)
fill_mat = tf.fill([4, 4], 10)
myzeros = tf.zeros([4, 4])
myones = tf.ones([4, 4])
myrand = tf.random_normal([4, 4], mean=0, stddev=1)
myrandu = tf.random_uniform([4, 4], minval=0, maxval=1)
myops = [const, fill_mat, myzeros, myones, myrand, myrandu]

with tf.Session() as sess:
    for op in myops:
        print(op.eval())
        print('\n')

a = tf.constant([[1, 2],
                 [3, 4]])

print(a.get_shape())
b = tf.constant([[10],
                 [100]])
print(b.get_shape())

with tf.Session() as sess:
    result = sess.run(tf.matmul(a, b))
    print(result)