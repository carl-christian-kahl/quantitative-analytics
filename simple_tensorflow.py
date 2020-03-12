import numpy as np
import tensorflow as tf

# Build a dataflow graph.
c = tf.constant(2.0)
d = tf.constant(3.0)
e = c*c*d

de = tf.gradients(ys=[e], xs=[c,d])

# Construct a `Session` to execute the graph.
sess = tf.compat.v1.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(de)

print(result)
