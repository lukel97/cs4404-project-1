import argparse
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
tfgan = tf.contrib.gan

tf.logging.set_verbosity(tf.logging.INFO)

def load_image(filepath):
    file = tf.read_file(filepath)
    img_decoded = tf.image.decode_jpeg(file, channels=3)
    return tfgan.eval.preprocess_image(img_decoded, 64, 64)

bn = lambda x: layers.batch_norm(x, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
def generator(noise, dim=64):
    net = layers.conv2d(noise, dim, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=bn)
    net = layers.conv2d(net, dim * 2, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=bn)

    def residual_cell(x, dim):
        y = layers.conv2d(x, dim, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=bn)
        return x + y

    for i in range(9):
        net = residual_cell(net, dim * 2)

    # transpose = deconv
    net = tf.image.resize_images(net, tf.cast([dim / 2, dim / 2], tf.int32))
    net = layers.conv2d_transpose(net, dim, 4, activation_fn=tf.nn.relu, normalizer_fn=bn)
    net = tf.image.resize_images(net, tf.cast([dim, dim], tf.int32))
    net = layers.conv2d_transpose(net, 3, 4, activation_fn=tf.nn.tanh)

    return net

def discriminator(img, generator_inputs, dim=64):
    net = layers.conv2d(img, dim, 4, 2)
    net = tf.nn.leaky_relu(net, alpha=0.2)

    net = layers.conv2d(net, dim * 2, 4, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=bn)
    net = layers.conv2d(net, dim * 4, 4, 2, activation_fn=tf.nn.leaky_relu, normalizer_fn=bn)
    net = layers.conv2d(net, dim * 8, 4, 1, activation_fn=tf.nn.leaky_relu, normalizer_fn=bn)
    net = layers.conv2d(net, 1, 4, 8)
    return tf.sigmoid(tf.squeeze(net))

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
      default=10)
  args = parser.parse_args()  
  
  batch_size = args.batch_size

  summer_filenames = tf.data.Dataset.list_files(os.path.join(args.data_dir, "trainA/*.jpg"))
  summer_images = summer_filenames.map(load_image).batch(batch_size).repeat()
  summer_iterator = summer_images.make_one_shot_iterator()
  
  winter_filenames = tf.data.Dataset.list_files(os.path.join(args.data_dir, "trainB/*.jpg"))
  winter_images = winter_filenames.map(load_image).batch(batch_size).repeat()
  winter_iterator = winter_images.make_one_shot_iterator()

  xs = summer_iterator.get_next()
  ys = winter_iterator.get_next()

  xs.set_shape([batch_size, None, None, None])
  ys.set_shape([batch_size, None, None, None])

  cyclegan_model = tfgan.cyclegan_model(generator, discriminator, xs, ys)
  tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

  cyclegan_loss = tfgan.cyclegan_loss(cyclegan_model)
  gen_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5)
  dis_opt = tf.train.AdamOptimizer(0.0001, beta1=0.5)
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

  tfgan.gan_train(train_ops,
                  args.job_dir,
                  hooks=[tf.train.StopAtStepHook(80000),
                         tf.train.LoggingTensorHook([status_message], every_n_iter=10)],
                  config=tf.ConfigProto(log_device_placement=True))

  
