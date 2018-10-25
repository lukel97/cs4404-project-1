import argparse
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.gan as tfgan

def load_image(filepath):
    file = tf.read_file(filepath)
    img_decoded = tf.image.decode_jpeg(file, channels=3)
    return tfgan.eval.preprocess_image(img_decoded, 256, 256)

bn = lambda x: layers.batch_norm(x, scale=True, decay=0.9, epsilon=1e-5, is_training=True, updates_collections=None)

def generator(noise, dim=64):
    paddings = lambda x: tf.constant([[0,0],[x,x],[x,x],[0,0]])

    net = tf.pad(noise, paddings(3), 'REFLECT')
    net = layers.conv2d(net, dim, 7, 1,
                        padding='VALID',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=bn,
                        biases_initializer=None)

    # downsampling
    net = layers.conv2d(net, dim * 2, 3, 2, activation_fn=tf.nn.relu, normalizer_fn=bn, biases_initializer=None)
    net = layers.conv2d(net, dim * 4, 3, 2, activation_fn=tf.nn.relu, normalizer_fn=bn, biases_initializer=None)

    # resnet
    def residual_cell(x, dim):
        with tf.variable_scope('rescell', reuse=tf.AUTO_REUSE):
            y = tf.pad(x, paddings(1), 'REFLECT')
            y = layers.conv2d(y, dim, 3, 1,
                              padding='VALID',
                              activation_fn=tf.nn.relu,
                              normalizer_fn=bn,
                              biases_initializer=None)
            y = tf.pad(y, paddings(1), 'REFLECT')
            y = layers.conv2d(y, dim, 3, 1, padding='VALID', normalizer_fn=bn)
            return x + y

    for i in range(9):
        net = residual_cell(net, dim * 4)

    # upsampling
    net = layers.conv2d_transpose(net, dim * 2, 3, 2,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=bn,
                                  biases_initializer=None)
    net = layers.conv2d_transpose(net, dim, 3, 2,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=bn,
                                  biases_initializer=None)

    net = tf.pad(net, paddings(3), 'REFLECT')
    net = layers.conv2d(net, 3, 7, 1, padding='VALID', activation_fn=tf.nn.tanh)

    # deconv with NxN to avoid checkerboard artifacts
    # net = tf.image.resize_images(net, tf.cast([dim / 2, dim / 2], tf.int32))
    # net = tf.image.resize_images(net, tf.cast([dim, dim], tf.int32))

    return net

def discriminator(img, generator_inputs, dim=64):

    net = layers.conv2d(img, dim, 4, 2, activation_fn=tf.nn.leaky_relu)

    net = layers.conv2d(net, dim * 2, 4, 2, 
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=bn,
                        biases_initializer=None)
    net = layers.conv2d(net, dim * 4, 4, 2,
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=bn,
                        biases_initializer=None)
    net = layers.conv2d(net, dim * 8, 4, 1,
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=bn,
                        biases_initializer=None)
    
    net = layers.conv2d(net, 1, 4, 1, activation_fn=None)
    return net

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      default='/tmp/cyclegan-logs')
  parser.add_argument(
      '--data-dir',
      help='Location to dataset containing trainA/trainB subdirectories',
      default='../datasets/summer2winter_yosemite')
  parser.add_argument(
      '-b',
      '--batch-size',
      type=int,
      help='Batch size',
      default=1)
  args = parser.parse_args()  
  
  batch_size = args.batch_size

  summer_filenames = tf.data.Dataset.list_files(os.path.join(args.data_dir, "trainA/*.jpg"))
  summer_images = summer_filenames.map(load_image).repeat(200).shuffle(1000).batch(batch_size)
  summer_iterator = summer_images.make_one_shot_iterator()
  
  winter_filenames = tf.data.Dataset.list_files(os.path.join(args.data_dir, "trainB/*.jpg"))
  winter_images = winter_filenames.map(load_image).repeat(200).shuffle(1000).batch(batch_size)
  winter_iterator = winter_images.make_one_shot_iterator()

  xs = summer_iterator.get_next()
  ys = winter_iterator.get_next()

  xs.set_shape([batch_size, None, None, None])
  ys.set_shape([batch_size, None, None, None])

  cyclegan_model = tfgan.cyclegan_model(generator, discriminator, xs, ys)
  tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

  cyclegan_loss = tfgan.cyclegan_loss(cyclegan_model,
                                      # speeds things up
                                      tensor_pool_fn=tf.contrib.gan.features.tensor_pool)
  gen_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5)
  dis_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5)
  
  train_ops = tfgan.gan_train_ops(cyclegan_model,
      cyclegan_loss,
      gen_opt,
      dis_opt)
  train_steps = tfgan.GANTrainSteps(1, 1)

  status_message = tf.string_join( [
            'Starting train step: ',
            tf.as_string(tf.train.get_or_create_global_step())
        ],
        name='status_message')

  tf.logging.set_verbosity(tf.logging.INFO)

  tfgan.gan_train(train_ops,
                  args.job_dir,
                  hooks=[tf.train.StopAtStepHook(500000),
                         tf.train.LoggingTensorHook([status_message], every_n_iter=10)])

  
