# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org_licenses_LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for SqueezeNet inference.

To run, use:
  bazel run -c opt --config=cuda \
      third_party_tensorflow_models_image_squeezenet:squeezenet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 400,
                            """Number of batches to run.""")


def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  """Build the AlexNet model.

  Args:
    images: Images Tensor

  Returns:
    pool10: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([7, 7, 3, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  # TODO(shlens, jiayq): Add a GPU version of local response normalization.

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # fire2_squeeze1x1
  with tf.name_scope('fire2_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 96, 16], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire2_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire2_squeeze1x1)

  # fire2_expand1x1
  with tf.name_scope('fire2_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 16, 64],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire2_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire2_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire2_expand1x1)

  # fire2_expand3x3
  with tf.name_scope('fire2_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 64],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire2_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire2_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire2_expand3x3)

  fire2_concat = tf.concat(3, [fire2_expand1x1, fire2_expand3x3])

  # fire3_squeeze1x1
  with tf.name_scope('fire3_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 128, 16], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire2_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire3_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire3_squeeze1x1)

  # fire3_expand1x1
  with tf.name_scope('fire3_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 16, 64],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire3_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire3_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire3_expand1x1)

  # fire3_expand3x3
  with tf.name_scope('fire3_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 64],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire3_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire3_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire3_expand3x3)

  fire3_concat = tf.concat(3, [fire3_expand1x1, fire3_expand3x3])

  # fire4_squeeze1x1
  with tf.name_scope('fire4_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 128, 32], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire3_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire4_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire4_squeeze1x1)

  # fire4_expand1x1
  with tf.name_scope('fire4_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 32, 128],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire4_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire4_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire4_expand1x1)

  # fire4_expand3x3
  with tf.name_scope('fire4_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 128],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire4_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire4_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire4_expand3x3)

  fire4_concat = tf.concat(3, [fire4_expand1x1, fire4_expand3x3])

  # pool4
  pool4 = tf.nn.max_pool(fire4_concat,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool4')
  print_activations(pool4)

  # fire5_squeeze1x1
  with tf.name_scope('fire5_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 256, 32], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire5_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire5_squeeze1x1)

  # fire5_expand1x1
  with tf.name_scope('fire5_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 32, 128],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire5_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire5_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire5_expand1x1)

  # fire5_expand3x3
  with tf.name_scope('fire5_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 128],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire5_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire5_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire5_expand3x3)

  fire5_concat = tf.concat(3, [fire5_expand1x1, fire5_expand3x3])

  # fire6_squeeze1x1
  with tf.name_scope('fire6_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 256, 48], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire5_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire6_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire6_squeeze1x1)

  # fire6_expand1x1
  with tf.name_scope('fire6_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 48, 192],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire6_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire6_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire6_expand1x1)

  # fire6_expand3x3
  with tf.name_scope('fire6_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 48, 192],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire6_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire6_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire6_expand3x3)

  fire6_concat = tf.concat(3, [fire6_expand1x1, fire6_expand3x3])

  # fire7_squeeze1x1
  with tf.name_scope('fire7_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 384, 48], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire6_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[48], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire7_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire7_squeeze1x1)

  # fire7_expand1x1
  with tf.name_scope('fire7_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 48, 192],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire7_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire7_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire7_expand1x1)

  # fire7_expand3x3
  with tf.name_scope('fire7_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 48, 192],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire7_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire7_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire7_expand3x3)

  fire7_concat = tf.concat(3, [fire7_expand1x1, fire7_expand3x3])

  # fire8_squeeze1x1
  with tf.name_scope('fire8_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 384, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire7_concat, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire8_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire8_squeeze1x1)

  # fire8_expand1x1
  with tf.name_scope('fire8_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 64, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire8_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire8_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire8_expand1x1)

  # fire8_expand3x3
  with tf.name_scope('fire8_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire8_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire8_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire8_expand3x3)

  fire8_concat = tf.concat(3, [fire8_expand1x1, fire8_expand3x3])

  # pool8
  pool8 = tf.nn.max_pool(fire8_concat,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool8')
  print_activations(pool8)

  # fire9_squeeze1x1
  with tf.name_scope('fire9_squeeze1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire9_squeeze1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(fire9_squeeze1x1)

  # fire9_expand1x1
  with tf.name_scope('fire9_expand1x1') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 64, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire9_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire9_expand1x1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire9_expand1x1)

  # fire9_expand3x3
  with tf.name_scope('fire9_expand3x3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(fire9_squeeze1x1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    fire9_expand3x3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(fire9_expand3x3)

  fire9_concat = tf.concat(3, [fire9_expand1x1, fire9_expand3x3])

  # drop9
  drop9 = tf.nn.dropout(fire9_concat, 0.5)

  # conv10
  with tf.name_scope('conv10') as scope:
    kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 1000],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(drop9, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv10 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv10)

  # pool10
  pool10 = tf.nn.avg_pool(conv10,
                         ksize=[1, conv10.get_shape()[1], conv10.get_shape()[2], 1],
                         strides=[1, 1, 1, 1],
                         padding='VALID',
                         name='pool10')
  print_activations(pool10)

  return pool10, parameters


def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pool10, parameters = inference(images)

    # Build an initialization operation.
    init = tf.initialize_all_variables()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, pool10, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(pool10)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")

    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
