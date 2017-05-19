wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
wget https://upload.wikimedia.org/wikipedia/commons/9/98/Sanzio_01_Plato_Aristotle.jpg
convert -crop 224x224+130+40 Sanzio_01_Plato_Aristotle.jpg crop.jpg