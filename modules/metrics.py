import keras.backend as K

def PSNR(y_true, y_pred):
  """
  PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  The equation is:
  PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

  Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
  """
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

def RGB_to_Y(image):
  R = image[:, :, :, 0]
  G = image[:, :, :, 1]
  B = image[:, :, :, 2]

  Y = 16 + (65.738 * R) + 129.057 * G + 25.064 * B
  return Y / 255.0

def PSNR_Y(y_true, y_pred):
  y_true = RGB_to_Y(y_true)
  y_pred = RGB_to_Y(y_pred)
  return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)