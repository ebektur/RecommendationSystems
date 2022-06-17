from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings 
from tensorflow.python.keras.backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import vgg16
import datetime
import traceback
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from rest_framework.decorators import api_view
from rest_framework.response import Response

import numpy as np
import tensorflow as tf
from keras.models import load_model
global graph,model
import time


#modeli basta load et
#creating a dictionary of classes
class_dict = {'Abstract Art': 0,
            'Abstract Expressionism': 1,
            'Art Informel': 2, 
            'Baroque': 3,
            'Color Field Painting': 4,
            'Cubism': 5,
            'Early Renaissance': 6,
            'Expressionism': 7,
            'High Renaissance': 8, 
            'Impressionism': 9,
            'Late Renaissance': 10,
            'Magic Realism': 11, 
            'Minimalism': 12, 
            'Modern': 13, 
            'Naive Art': 14, 
            'Neoclassicism': 15, 
            'Northern Renaissance': 16,
            'Pop Art': 17,
            'Post-Impressionism': 18,
            'Realism': 19, 
            'Rococo': 20, 
            'Romanticism': 21,
            'Surrealism': 22,
            'Symbolism': 23, 
            'Ukiyo-e': 24}

def index(request):
    return render(request,'success.html')


# def index(request):
#     if  request.method == "POST":
#         f=request.FILES['sentFile'] # here you get the files needed
#         response = {}
#         file_name = "pic.jpg"
#         file_name_2 = default_storage.save(file_name, f)
#         file_url = default_storage.url(file_name_2)
#         #timestamp, input(filename), output in database
#         start = time.time()
#         original = load_img(file_url, target_size=(224, 224))
#         end = time.time()
#         time_load_img = end - start

#         numpy_image = img_to_array(original)
        

#         image_batch = np.expand_dims(numpy_image, axis=0)
#         # prepare the image for the VGG model
#         #timer
#         start = time.time()
#         processed_image = vgg16.preprocess_input(image_batch.copy())
#         end = time.time()
#         time_vgg_preprocessing = end - start

#         # get the predicted probabilities for each class
#         #with settings.GRAPH1.as_default():
#             #timer
#         start = time.time()
#         VGG_MODEL = vgg16.VGG16(weights="imagenet")
#         end = time.time()
#         time_load_model_vgg = end - start

#         print("graph1 works")
#         #set_session(settings.SESS)
#         predictions=VGG_MODEL.predict(processed_image)
#         label = decode_predictions(predictions)
#         label = list(label)[0]
#         response['name'] = str(label)
#         response['type'] = "loaded in views.py"
#         response['time_load_img'] = str(time_load_img)
#         response['time_vgg_preprocessing'] = str(time_vgg_preprocessing)
#         response['time_load_model_vgg'] = str(time_load_model_vgg)

#         return render(request,'homepage.html',response)
#     else:
#         return render(request,'homepage.html')

def genre(request):
    if  request.method == "POST":
        f=request.FILES['sentFile'] # here you get the files needed
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        start = time.time()
        original = load_img(file_url, target_size=(224, 224))
        end = time.time()
        time_load_img = end-start
        numpy_image = img_to_array(original)
        numpy_image = numpy_image / 255
        image_batch = np.expand_dims(numpy_image, axis=0)
        # prepare the image for the VGG model
        start = time.time()
        model_vgg16_1 = vgg16.VGG16(include_top=False, weights='imagenet') 
        end = time.time()
        time_load_model_vgg = end-start

        bottleneck_prediction = model_vgg16_1.predict(image_batch) 

        # get the predicted probabilities for each class
        
        start = time.time()
        model_artstyle = load_model('predictors/models/my_model_25.h5')
        end = time.time()
        time_load_model_genre = end-start
        #setsession(settings.SESS)
        class_predicted = model_artstyle.predict(bottleneck_prediction)
        inID = class_predicted[0]
        index_max = np.argmax(inID)
        inv_map = {v: k for k, v in class_dict.items()}  
        label = inv_map[index_max] 
        response['genre'] = str(label)
        response['type'] = "loaded in views.py"
        response['time_load_img'] = str(time_load_img)
        response['time_load_model_genre'] = str(time_load_model_genre)
        response['time_load_model_vgg'] = str(time_load_model_vgg)

        return render(request, 'genre.html', response)
    else:
        return render(request,'genre.html')