#GPU vs CPU
'''
In this notebook we learn how to run a program on different devices.

There are usually multiple computing devices available on each machine. TensorFlow, supports three types of devices:

    "/cpu:0": The CPU of your machine.
    "/gpu:0": The GPU of your machine, if you have one.
    "/device:XLA:0": Optimized domain-specific compiler for linear algebra.
'''
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.log_device_placement=True

#List of CPU and GPUs---------------------------------------------------------------------------
#How to get list of CPU and GPUs on your machine ?
from tensorflow.python.client import device_lib

def get_available_gpus():
    with tf.Session(config=config) as sess:
        local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
get_available_gpus()

#Logging device----------------------------------------------------------------------------
'''
It is recommended to use logging device placement when using GPUs, as this lets you easily debug issues relating to different device usage. 
This prints the usage of devices to the log, allowing you to see when devices change and how that affects the graph. 
Unfortunately, the current version of Jupyter notebook does not show the logs properly, 
but still you can print out the nodes as a json file and check the devices. It will work fine if your script is running outside of Jupyter notebook though.
'''
#You can see that a, b and c are all run on GPU0.
def print_logging_device():
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.constant([1.0, 2.0, 3.0, 4.0, ], shape=[2, 2], name='c')
    mu = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    with tf.Session(config=config) as sess:
        # Runs the op.
        options = tf.RunOptions(output_partition_graphs=True)
        metadata = tf.RunMetadata()
        c_val = sess.run(mu, options=options, run_metadata=metadata)
        print (c_val)
        print (metadata.partition_graphs)

print_logging_device()

#Device placement:--------------------------------------------------------------------------
'''
As mentioned, by default, GPU get priority for the operations which support running on it. 
But, you can manually place an operation on a device to be run. You can use tf.device to assign the operations to a device context.
'''
# Creates a new graph.
myGraph = tf.Graph()
with tf.Session(config=config, graph=myGraph) as sess:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.constant([1.0, 2.0, 3.0, 4.0, ], shape=[2, 2], name='c')
        mu = tf.matmul(a, b)
    with tf.device('/cpu:0'):
        ad = tf.add(mu,c)
    print (sess.run(ad))


#Lets use one of Alex Mordvintsev deep dream notebook to visualize the above graph. You can change the color to see the operations assigned to GPU and CPU.
# Helper functions for TF Graph visualization
from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

graph_def = myGraph.as_graph_def()
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)

#Multiplication on GPU and CPU---------------------------------------------------------------------------------------
#In the following cell, we define a function, to measure the speed of matrix multiplication in CPU and GPU.
def matrix_mul(device_name, matrix_sizes):
    time_values = []
    #device_name = "/cpu:0"
    for size in matrix_sizes:
        with tf.device(device_name):
            random_matrix = tf.random_uniform(shape=(1000,1000), minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        with tf.Session(config=config) as session:
            startTime = datetime.now()
            result = session.run(sum_operation)
        td = datetime.now() - startTime
        time_values.append(td.microseconds/1000)
        print ("matrix shape:" + str(size) + "  --"+ device_name +" time: "+str(td.microseconds/1000))
    return time_values


matrix_sizes = range(100,2000,100)
time_values_gpu = matrix_mul("/gpu:0", matrix_sizes)
time_values_cpu = matrix_mul("/cpu:0", matrix_sizes)
print ("GPU time" +  str(time_values_gpu))
print ("CPU time" + str(time_values_cpu))


#Lets plot the performance of training the model on a CPU versus on a GPU.
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.plot(matrix_sizes[:len(time_values_gpu)], time_values_gpu, label='GPU')
plt.plot(matrix_sizes[:len(time_values_cpu)], time_values_cpu, label='CPU')
plt.ylabel('Time (sec)')
plt.xlabel('Size of Matrix ')
plt.legend(loc='best')
plt.show()

%%javascript
// Shutdown kernel
Jupyter.notebook.session.delete()