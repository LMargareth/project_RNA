# -----------------------------------------------------------
# Cargando modelo de disco
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
# dimensions of our images.

json_file = open('./model/cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model/cnn_model.h5")
print("Cargando modelo desde el disco ...")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Modelo cargado de disco!")

# Testing con una foto
test_image_path = './dataset/samples/8.jpg'
test_image = image.load_img(test_image_path,target_size = (50, 50))
plt.imshow(test_image)
plt.show()

#test_image se define el tamaño de la imagen para este caso es 50x50
test_image = image.load_img(test_image_path,target_size = (50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
if result[0][0] == 1:
    print(result[0][0], ' --> Es un perro')
else:
    print(result[0][0], ' --> Es un gato ')
