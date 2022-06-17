from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras.applications import vgg16
import os

class genreModelConfig(AppConfig):
    name = 'genreAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "my_model_25.h5")
    model = keras.models.load_model(MODEL_FILE)


class vggModelConfig(AppConfig):
    name = 'vggAPI'
    model = vgg16.VGG16(weights='imagenet')
 

class vggBaseModelConfig(AppConfig):
    name = 'vggAPI'
    model = vgg16.VGG16(include_top=False, weights='imagenet')
