# -----------------------------
# PARTE I - PREPARACION DE DATA
# -----------------------------
import datetime
print('Iniciando a las: ', datetime.datetime.now())
print("...")
#tiempo de demora se debe guardar tb para la tesis
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
#definiendo parametros iniciales
# rescale: Normalizar valores
# shear_range: Intensidad de corte (ángulo de corte en sentido antihorario en grados)  
# zoom_range: Rango para zoom aleatorio.se debe poner por buenas practicas
# horizontal_flip: valor booleno. Voltea aleatoriamente las entradas horizontalmente.

#test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (50, 50),
                                                 batch_size = 50,
                                                 class_mode = 'binary')
#esta etapa se carga las fotos
#flowfromdirectory identifica las carpetas como clase
# target_size: Redmiensionar la imagen a 50x50
# batch_size: Tamaño de los lotes batchs (Default: 10) toma 30 en 30 observaciones para actualizar los pesos
# class_mode: "binary" label tipo binario 1D

print(training_set.class_indices)
#muestra las clases 
# --------------------------------------
# PARTE II - RED NEURONAL CONVOLUCIONAL
# --------------------------------------
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializando la red neuronal CNN
cnn_model = Sequential()

# 1ra Capa Convolucional
# Convolution
# input: 50x50 imagenes con 3 canales -> (50, 50, 3) tensores.
# 32 filtros de 3x3 va usar 32 fitros de 3x3. 50-3+1=48*48 se aplica RelU no cambia
#input_shape esta recibiendo imagenes de tamaño 50x50 de 3 canales
cnn_model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (50, 50, 3)))
# Pooling, aqui se puede definir el stride=24*24
#pooling no disminuye la cantidad de imagenes aqui cambia la dimension
#24-3+1=22 la mtriz seria 11 por q se reduce
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

# 2da Capa Convolutional
# Convolution
cnn_model.add(Conv2D(32, (3, 3), activation = 'relu'))
# Pooling
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))

#La matriz obtenida finalmente se pasa a un vector
# Flattening
cnn_model.add(Flatten())

# Full connection | Clas
cnn_model.add(Dense(units = 128, activation = 'relu'))
cnn_model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
cnn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamiento
cnn_model.fit_generator(training_set,
                        steps_per_epoch = 3000,
                        epochs = 5)

# steps_per_epoch: (25000/30) Número total de pasos (batches) para una época.de los 25000agarra 30 en 30 y repites 2000 veces para actualizar los pesos
# epochs: Número total de época 5 veces pasa toda la data

# Guardar el modelo en disco
cnn_model_json = cnn_model.to_json()
with open("./model/cnn_model.json","w") as json_file:
  json_file.write(cnn_model_json)

cnn_model.save_weights("./model/cnn_model.h5")
print("Modelo guardado en disco ...")
print("...")
print('Terminando a las: ', datetime.datetime.now())

#-------------------------------------
# PARTE III: USAR EL MODELO ENTRENADO
#-------------------------------------
'''
# Cargando modelo de disco
from keras.models import model_from_json
import numpy as np

json_file = open('./model/cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model/cnn_model.h5")
print("Cargando modelo desde el disco ...")
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Modelo cargado de disco!")

# Testing con una foto
from keras.preprocessing import image
import matplotlib.pyplot as plt

test_image_path = './dataset/test/1.jpg'
test_image = image.load_img(test_image_path)
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

#mejoras la clasificacion sin reducir la imagen
#cambias las epocas
#tarea superar el accuracy
'''
    