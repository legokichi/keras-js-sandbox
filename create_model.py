if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='keras.js imagenet model converter')
    parser.add_argument("--model", action='store', type=str, default="", help='resnet50 | inception_v3 | vgg16 | xception | squeezenet')
    parser.add_argument("--compress", action='store_true', help='use https://github.com/nico-opendata/keras_compressor')
    args = parser.parse_args()

    from keras.backend import set_image_data_format, set_floatx, floatx
    # keras.backend.backend()
    # keras.backend.set_epsilon(1e-07)
    # keras.backend.epsilon()
    # set_floatx('float16')
    set_image_data_format("channels_last") # tensorflow

    if args.model == "resnet50":
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    elif args.model == "inception_v3":
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3))
    elif args.model == "xception":
        from keras.applications.xception import Xception
        model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))
    elif args.model == "vgg16":
        from keras.applications.vgg16 import VGG16
        model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    elif args.model == "squeezenet":
        import sys
        sys.path.append("./keras-squeezenet")
        from keras_squeezenet import SqueezeNet
        model = SqueezeNet(input_tensor=None, input_shape=None, weights='imagenet', classes=1000)
    else:
        exit()

    if args.compress == True:
        model.save("model_original.hdf5")
        import sys
        sys.path.append("./keras_compressor/keras_compressor")
        import subprocess
        subprocess.call("python ./keras_compressor/bin/keras-compressor.py model_original.hdf5 model_compressed.hdf5 --log-level DEBUG", shell=True)
        from keras.models import load_model
        model = load_model("model_compressed.hdf5")
        model.summary()
        model.save_weights("model.hdf5")

    model.summary()
    model.save_weights("model.hdf5")
    with open('model.json', 'w') as f: f.write(model.to_json())

    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    import subprocess
    subprocess.call("python ./keras-js/encoder.py model.hdf5", shell=True)

    exit()
