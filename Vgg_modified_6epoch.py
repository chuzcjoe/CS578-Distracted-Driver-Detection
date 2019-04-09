from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras import layers
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import ModelCheckpoint

def create_model():
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
  
  return model
  


def model_train():
  model = create_model()
  sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  checkpointer = ModelCheckpoint(filepath="weights.{epoch:04d}.hdf5", verbose=1, save_best_only=True, period=1)
  
  image_size = (224, 224)
  gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
  train_generator = gen.flow_from_directory("mask_rcnn_full", image_size, shuffle=True, batch_size=32, subset='training')
  validation_generator = gen.flow_from_directory("mask_rcnn_full", image_size, shuffle=True, batch_size=32, subset='validation')
  nb_val = len(validation_generator.filenames)
  
  model.fit_generator(train_generator, validation_data=validation_generator, nb_epoch=6, steps_per_epoch=400, validation_steps=nb_val//32,callbacks=[checkpointer])

def load_model(weights_path):
  model = create_model()
  model.load_weights(weights_path)
  return model
  

model_train()


VGG_epoch1 = load_model('weights.0001.hdf5')
VGG_epoch2 = load_model('weights.0002.hdf5')
VGG_epoch3 = load_model('weights.0003.hdf5')
VGG_epoch4 = load_model('weights.0004.hdf5')
VGG_epoch5 = load_model('weights.0005.hdf5')
VGG_epoch6 = load_model('weights.0006.hdf5')



from keras.preprocessing.image import *
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import ModelCheckpoint

def predict_on_test(test_dir):
  
  models = ["VGG_epoch1","VGG_epoch2","VGG_epoch3","VGG_epoch4","VGG_epoch5","VGG_epoch6"]
  test_datagen = ImageDataGenerator(rescale=1./255)
  height = 224
  width = 224
  batch_size = 32
  f = h5py.File('VGGpreds.h5', 'w')

  for model in models:
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(height, width),batch_size=1,class_mode='categorical', shuffle=False,)
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    preds = VGG_epoch1.predict_generator(test_generator, steps = nb_samples//batch_size)
    f.create_dataset(model[-6:],data=preds)
  
  f.close()
  