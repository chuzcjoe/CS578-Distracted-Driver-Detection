from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras import layers

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
    
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
# Create the model
model = Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(10, activation='softmax'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(layers.Convolution2D(512, (7, 7)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(layers.Convolution2D(512,(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(layers.Convolution2D(10,(1,1)))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation='softmax'))

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Show a summary of the model. Check the number of trainable parameters
model.summary()

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

image_size = (224, 224)
gen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
train_generator = gen.flow_from_directory("mask_rcnn_full", image_size, shuffle=True, batch_size=32, subset='training')
validation_generator = gen.flow_from_directory("mask_rcnn_full", image_size, shuffle=True, batch_size=32, subset='validation')

model.fit_generator(train_generator, validation_data=validation_generator, nb_epoch=6, steps_per_epoch=490, validation_steps=160)

model.save('testfit.h5')