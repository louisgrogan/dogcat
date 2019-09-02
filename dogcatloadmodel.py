from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

json_file = open('/content/drive/My Drive/cnn/dogcatmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("/content/drive/My Drive/cnn/dogcatmodel.h5")
print("Loaded model from disk")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/content/drive/My Drive/cnn/training_set/cats/cat.1064.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'
print(prediction)