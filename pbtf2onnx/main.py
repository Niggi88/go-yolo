import tensorflow as tf
graph_def = tf.GraphDef()
with tf.gfile.GFile("model.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
print([n.name for n in graph_def.node])