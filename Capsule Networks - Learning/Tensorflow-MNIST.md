

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
```


```python
print('TensorFlow version: ', tf.__version__)
```

    TensorFlow version:  1.11.0
    


```python
batch_size = 128
test_size = 256
```


```python
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
```


```python
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
```


```python
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
```

    WARNING:tensorflow:From <ipython-input-18-cdb85efb4396>:1: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use urllib or similar directly.
    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From C:\Users\Pragyaditya Das\Anaconda3\lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    


```python
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
```


```python
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

```


```python
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
```


```python
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

```

    WARNING:tensorflow:From <ipython-input-22-29c24a28cfe8>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
```

    0 0.96484375
    1 0.9765625
    2 0.99609375
    3 0.9921875
    4 0.99609375
    5 0.99609375
    6 0.984375
    7 0.9921875
    8 0.98828125
    9 0.99609375
    10 0.9921875
    11 0.9921875
    12 0.9921875
    13 0.98828125
    14 0.99609375
    15 1.0
    16 0.99609375
    17 0.9921875
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-24-e5947fbc655d> in <module>()
          8         for start, end in training_batch:
          9             sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
    ---> 10                                           p_keep_conv: 0.8, p_keep_hidden: 0.5})
         11 
         12         test_indices = np.arange(len(teX)) # Get A Test Batch
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in run(self, fetches, feed_dict, options, run_metadata)
        885     try:
        886       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 887                          run_metadata_ptr)
        888       if run_metadata:
        889         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1108     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1109       results = self._do_run(handle, final_targets, final_fetches,
    -> 1110                              feed_dict_tensor, options, run_metadata)
       1111     else:
       1112       results = []
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1284     if handle is None:
       1285       return self._do_call(_run_fn, feeds, fetches, targets, options,
    -> 1286                            run_metadata)
       1287     else:
       1288       return self._do_call(_prun_fn, handle, feeds, fetches)
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
       1290   def _do_call(self, fn, *args):
       1291     try:
    -> 1292       return fn(*args)
       1293     except errors.OpError as e:
       1294       message = compat.as_text(e.message)
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _run_fn(feed_dict, fetch_list, target_list, options, run_metadata)
       1275       self._extend_graph()
       1276       return self._call_tf_sessionrun(
    -> 1277           options, feed_dict, fetch_list, target_list, run_metadata)
       1278 
       1279     def _prun_fn(handle, feed_dict, fetch_list):
    

    ~\Anaconda3\lib\site-packages\tensorflow\python\client\session.py in _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list, run_metadata)
       1365     return tf_session.TF_SessionRun_wrapper(
       1366         self._session, options, feed_dict, fetch_list, target_list,
    -> 1367         run_metadata)
       1368 
       1369   def _call_tf_sessionprun(self, handle, feed_dict, fetch_list):
    

    KeyboardInterrupt: 

