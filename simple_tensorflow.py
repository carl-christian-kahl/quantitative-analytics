import numpy as np
import tensorflow as tf

_SQRT_2 = np.sqrt(2.0, dtype=np.float64)

def _ncdf(x):
  return (tf.math.erf(x / _SQRT_2) + 1) / 2

def _black_scholes(f,k,s,t):
    sdt = tf.math.sqrt(t) * s
    d1 = tf.math.log(f/k) / sdt + 0.5 * sdt
    return f*_ncdf(d1) - k*_ncdf(d1 - sdt)

# Build a dataflow graph.
f = tf.Variable(1.0)
k = tf. constant(1.0)
s = tf.constant(0.2)
t = tf.constant(1.0)

v = _black_scholes(f,k,s,t)
dv = tf.gradients(ys=[v], xs=[f,k,s,t])

current_graph = tf.compat.v1.get_default_graph()

#all_names = [op.name for op in current_graph.get_operations()]
all_names = [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='Variable']


print(all_names)

# Construct a `Session` to execute the graph.
sess = tf.compat.v1.Session()

# Execute the graph and store the value that `e` represents in `result`.
#result = sess.run(dv)

#print(result)
