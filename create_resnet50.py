from keras.backend import set_image_data_format, set_floatx, floatx
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# set_floatx('float16')
set_image_data_format("channels_last") # tensorflow


if __name__ == '__main__':
  from keras.applications.resnet50 import ResNet50

  model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
  
  import util_model
  util_model.convert(model)


  exit()
