import tensorflow as tf
from keras.layers import Lambda

subpixel_index = 0
def SubpixelConv2D(scale=2):
  global subpixel_index

  def subpixel_shape(input_shape):
    dims = [input_shape[0],
            None if input_shape[1] is None else input_shape[1] * scale,
            None if input_shape[2] is None else input_shape[2] * scale,
            int(input_shape[3] / (scale ** 2))]
    output_shape = tuple(dims)
    return output_shape

  def subpixel(x):
    return tf.nn.depth_to_space(x, scale)

  subpixel_index += 1
  return Lambda(subpixel, output_shape=subpixel_shape, name=f"subpixel_conv2d_{subpixel_index}")