import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os

# For newer TensorFlow versions (2.x), use:
graph_def = tf.compat.v1.GraphDef()   # instead of tf.GraphDef()

# Or alternatively use this full script:
import tensorflow as tf

def inspect_pb(pb_path):
    print(pb_path, os.path.exists(pb_path))
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
    print("Nodes in the graph:")
    for node in graph_def.node:
        print(node.name)
    

inspect_pb("./models/lower_cart_empty_loaded.pb")