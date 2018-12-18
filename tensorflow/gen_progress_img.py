import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from main import generator, discriminator

def load_image(filepath):
    file = tf.read_file(filepath)
    img_decoded = tf.image.decode_jpeg(file, channels=3)
    return tfgan.eval.preprocess_image(img_decoded, 256, 256)


batch_size = 1231
num_epochs = 50

for checkpoint in range(batch_size * 2, batch_size * num_epochs + 1, batch_size * 2):
    tf.reset_default_graph()
    
    real_img = 'pytorch_real_images/epoch' + str(checkpoint // batch_size).zfill(3) + '_real_B.png'
    
    with tf.Session() as sess:

        def write_img(path, img):
            scaled_img = tf.cast(((img + 1.0) / 2.0) * 255, tf.uint8)

            saver = tf.train.Saver()
            saver.restore(sess, 'gs://cs4404-a1-mlengine/cyclegan_summer2winter_yosemite_100_epochs/persistent_checkpoints/model.ckpt-' + str(checkpoint))

            sess.run(tf.write_file(path, tf.image.encode_jpeg(scaled_img)))

        def convert_to_summer(path):
            with tf.variable_scope('ModelY2X'):
                with tf.variable_scope('Generator'):
                    img = load_image(path)
                    return generator([img])
                    
        
        res = convert_to_summer(real_img)
        write_img('progress_epoch' + str(checkpoint // batch_size).zfill(3) + '.jpg', res[0])
