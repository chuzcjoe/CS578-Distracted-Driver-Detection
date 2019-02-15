from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer
from keras.optimizers import SGD


def get_model():
    # initialize vgg16 model
    vgg_model = VGG16(input_shape=(224, 224, 3))
    # Remove the output layer
    vgg_model.layers.pop()
    # Construct input layer
    inp = InputLayer(input_shape=(224, 224, 3))
    # Define customized sequential layer
    model = Sequential()
    model.add(inp)
    # Add all the layers in vgg model
    for l in vgg_model.layers:
        model.add(l)
    # Add customizied output layer
    model.add(Dense(10, activation='softmax'))
    return model


def run_training():
    model = get_model()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=10, epochs=15,
                        validation_data=train_batches, validation_steps=10, verbose=2)


TRAIN_PATH = '/content/train/'

# Define training_batches
train_batches = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2).flow_from_directory(TRAIN_PATH, target_size=(
    224, 224), classes=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], batch_size=10, class_mode="categorical", shuffle=True)

# Valid_batches = ImageDataGenerator().flow_from_directory(TEST_PATH, target_size=(224, 224), classes=['c0, c1, c2, c3, c4, c5, c6, c7, c8, c9'], batch_size=10, class_mode="categorical", shuffle=True)
run_training()
