import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import models,layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

class Model:
    
    def resize_and_rescale(self, image):
        # Function to resize and rescale the input image
        image = tf.image.resize(image, (224, 224))
        image /= 255.0
        return image
        
    def custom_model(self, train_data, validation_data):
        self.model = Sequential()

        self.model.add(tf.keras.layers.Lambda(self.resize_and_rescale, input_shape=(180, 180, 3)))
        
        self.model.add(Conv2D(60, (3,3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(2,2))

        self.model.add(Conv2D(32, (3,3), padding='same', activation='relu'))  
        self.model.add(MaxPooling2D(2,2))

        self.model.add(Conv2D(32, (3,3), padding='same', activation='relu'))  
        self.model.add(MaxPooling2D(2,2))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        return self.model
    
    def pre_trained_model(self, pre_trained, train_data, validation_data):
        # Load pre-trained model with imagenet weights
        base_model =  pre_trained(include_top=False, weights = 'imagenet', input_shape=(180, 180, 3), pooling='avg')

        # Freeze the weights of the pre-trained layers in the model
        base_model.trainable = False

        # Add new layers on top of the model
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model