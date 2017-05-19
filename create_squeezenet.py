from keras.backend import set_image_data_format, set_floatx, floatx
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# set_floatx('float16')
set_image_data_format("channels_last") # tensorflow

if __name__ == '__main__':
  import subprocess
  subprocess.call("cd squeezenet_demo; git reset --hard; patch -p1 < squeezenet_demo.patch", shell=True)

  from squeezenet_demo.model import SqueezeNet

  model = SqueezeNet(nb_classes=100)
  
  import util_model
  util_model.convert(model)

  exit()