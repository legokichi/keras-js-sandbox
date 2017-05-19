
if __name__ == '__main__':
  from keras.applications.vgg16 import VGG16

  model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
  model.summary()
  model.save_weights("model.hdf5")
  with open('model.json', 'w') as f: f.write(model.to_json())

  from keras.utils import plot_model
  plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

  import subprocess
  subprocess.call("python ./node_modules/keras-js/encoder.py model.hdf5", shell=True) 

  exit()