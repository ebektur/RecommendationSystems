from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from .apps import *
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
import logging
from .forms import *
from .models import *
from django.http import HttpResponseRedirect
from formtools.wizard.views import SessionWizardView
import json
import io
import csv
import operator
import functools
from django.db.models import Q
import scipy
from scipy.spatial import KDTree
from sklearn.preprocessing import minmax_scale
from scipy.special import softmax


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


def genre_preload2(request):
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
        model_vgg16_1 = vggBaseModelConfig.model
        end = time.time()
        time_load_model_vgg = end-start

        bottleneck_prediction = model_vgg16_1.predict(image_batch)
        
        start = time.time()
        model_artstyle = genreModelConfig.model
        end = time.time()
        time_load_model_genre = end-start

        class_predicted = model_artstyle.predict(bottleneck_prediction)
        inID = class_predicted[0]
        index_max = np.argmax(inID)
        inv_map = {v: k for k, v in class_dict.items()}
        label = inv_map[index_max] 
        response['genre'] = str(label)
        response['type'] = "loaded in apps.py"
        response['time_load_img'] = str(time_load_img)
        response['time_load_model_genre'] = str(time_load_model_genre)
        response['time_load_model_vgg'] = str(time_load_model_vgg)
        return render(request, 'genre.html', response)
    else:
    	return render(request,'genre.html')

def vgg_model_view(request):
    if  request.method == "POST":
        f=request.FILES['sentFile'] # here you get the files needed
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        #timestamp, input(filename), output in database
        start = time.time()
        original = load_img(file_url, target_size=(224, 224))
        end = time.time()
        time_load_img = end - start

        numpy_image = img_to_array(original)
        

        image_batch = np.expand_dims(numpy_image, axis=0)
        # prepare the image for the VGG model
        #timer
        start = time.time()
        processed_image = vgg16.preprocess_input(image_batch.copy())
        end = time.time()
        time_vgg_preprocessing = end - start

        # get the predicted probabil
            #timer
        start = time.time()
        VGG_MODEL = vggModelConfig.model
        end = time.time()
        time_load_model_vgg = end - start

        print("graph1 works")
        #set_session(settings.SESS)
        predictions=VGG_MODEL.predict(processed_image)
        label = decode_predictions(predictions)
        label = list(label)[0]
        response['name'] = str(label)
        response['type'] = "loaded in views.py"
        response['time_load_img'] = str(time_load_img)
        response['time_vgg_preprocessing'] = str(time_vgg_preprocessing)
        response['time_load_model_vgg'] = str(time_load_model_vgg)

        return render(request,'homepage.html',response)
    else:
        return render(request,'homepage.html')

def success(request):
    return render(request,'success.html')

def genre_test_multiple(request):
	if request.method == 'POST':
		form = MultipleChoiceGenreForm(request.POST)
		this_test = Test()
		this_test.save()
		if form.is_valid():
			selected_genres = form.cleaned_data.get('picked') #"x-xx-xxx"
			this_question, created = GenreTestNew.objects.get_or_create(test_id=this_test)
			this_question.save()
			for selected_genre in selected_genres:
				one_answer, created = Answer.objects.get_or_create(answer=selected_genre, genre_test=this_question)
				one_answer.save()
		return HttpResponseRedirect('success')
	else:
		form = MultipleChoiceGenreForm
	return render(request, "genre_multiple.html", {
        "form": form
    })

def object_test_multiple(request):
	if request.method == 'POST':
		form = MultipleChoiceObjectForm(request.POST)
		this_test = Test() #put them under the same test
		this_test.save()
		if form.is_valid():
			selected_genres = form.cleaned_data.get('picked_genre') #"x-xx-xxx"
			this_question, created = GenreTestNew.objects.get_or_create(test_id=this_test)
			this_question.save()
			for selected_genre in selected_genres:
				one_answer, created = Answer.objects.get_or_create(answer=selected_genre, genre_test=this_question)
				one_answer.save()
		return HttpResponseRedirect('success')
	else:
		form = MultipleChoiceObjectForm
	return render(request, "object_multiple.html", {
        "form": form
    })

def survey(request):
	return render(request, 'survey/home.html')
def question(request):
    return render(request, 'survey/survey.html')    
def question_save(request):
    if request.method!="POST":
        return render(request, 'survey/survey.html') 
    else:
        rangeInput = request.POST.get("rangeInput")
        colorquestion=request.POST.get("colorquestion[]")
        #nested dictionary
        #survey_artworks_weights = {artwork_name_1: {'score_from_colors_coef_and_user_input': '90', 'score_from_objects_coef_and_user_input': '27', 'score_from_genre_coef_and_user_input': '1'},
        #artwork_name_2: {'score_from_colors_coef_and_user_input': '80', 'score_from_objects_coef_and_user_input': '17', 'score_from_genre_coef_and_user_input': '0'}
        #}
        #bi de total_Score var

        survey_artworks_weights = {}
        #color_question.request.POST['']
        colorquestion = colorquestion.lstrip('#')
        lv = len(colorquestion)
        colorquestion = list(int(colorquestion[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

        requested_color = []
        #query = functools.reduce(operator.or_,[Q(color_one__contains=query_obj) for query_obj in colorquestion])
        results = ColorOption.objects.values_list("colors1_rgb_list", "colors2_rgb_list", "colors3_rgb_list")
        clean_results = []
        for i in range(0, len(results)):
            for j in range(0, len(results[i])):
                clean_results.append(list(results[i][j].strip('][').split(', ')))
        #logging.error(clean_results)
        for k in range(0, len(clean_results)):
            for m in range(0, len(clean_results[k])):
                clean_results[k][m] = int(clean_results[k][m])
        ##now use knn model to find closest ones
        np_array_data = np.array(clean_results)
        tree= KDTree(np_array_data)
        dist, idx2 = tree.query(colorquestion, 10)
        closest_tree_colors= np_array_data[idx2]
        #closest_tree_colors = list(closest_tree_colors)
        ## search in db with of the closest colors
        #query = functools.reduce(operator.or_,[Q(color_one__contains=closest_tree_colors[0])])
        found_colors_list = []
        found_colors_urls = []
        for i in range(0, len(closest_tree_colors)):
            found_colors = ColorOption.objects.filter(Q(colors1_rgb_list= str(list(closest_tree_colors[i]))) | Q(colors2_rgb_list= str(list(closest_tree_colors[i])))
                | Q(colors3_rgb_list= str(list(closest_tree_colors[i]))))
            if ColorOption.objects.filter(Q(colors1_rgb_list= str(list(closest_tree_colors[i])))).exists():
                color_coef = found_colors[0].color_one_coefficient
            if ColorOption.objects.filter(Q(colors2_rgb_list= str(list(closest_tree_colors[i])))).exists():
                color_coef = found_colors[0].color_two_coefficient
            if ColorOption.objects.filter(Q(colors3_rgb_list= str(list(closest_tree_colors[i])))).exists():
                color_coef = found_colors[0].color_three_coefficient
            found_artwork_colors = found_colors.values_list('artwork_name', flat=True)

            found_colors_url = ColorOption.objects.filter(Q(colors1_rgb_list= str(list(closest_tree_colors[i]))) | Q(colors2_rgb_list= str(list(closest_tree_colors[i])))
                | Q(colors3_rgb_list= str(list(closest_tree_colors[i])))).values_list('url', flat=True)
            found_artwork_colors = list(found_artwork_colors)
            found_colors_url = list(found_colors_url)
            found_colors_urls = found_colors_url + found_colors_urls
            found_colors_list = found_colors_list + found_artwork_colors #artwork-urls
        requested_colors = np.array(found_colors_list)
        requested_colors = np.unique(found_colors_list)
        requested_colors = list(found_colors_list) # artwork-urls
        for i in range(len(requested_colors)):
            #add

            if str(requested_colors[i]) in survey_artworks_weights:
                #dictionary already exists
                survey_artworks_weights[str(requested_colors[i])] = int(color_coef * 100.0) * int(rangeInput)
                #survey_artworks_weights[str(requested_colors[i])] = int(survey_artworks_weights[str(requested_colors[i])]) + int(rangeInput)
                #add the weight 
            else:
                #add the key
                survey_artworks_weights[str(requested_colors[i])] = {}
                survey_artworks_weights[str(requested_colors[i])]['score_from_objects_coef_and_user_input'] = 0
                survey_artworks_weights[str(requested_colors[i])]['total_score'] = 0
                survey_artworks_weights[str(requested_colors[i])]['score_from_colors_coef_and_user_input'] = int(color_coef * 100.0) * int(rangeInput)
                survey_artworks_weights[str(requested_colors[i])]['chosen_artist'] = 0
                #survey_artworks_weights[str(requested_colors[i])] = int(color_coef) * int(rangeInput)

        request.session['requested_color'] = found_colors_list
        request.session['requested_color_urls'] = found_colors_urls
        request.session['survey_artworks_weights'] = survey_artworks_weights
        #logging.error(found_colors_list)

    try:
        this_test = Test()
        this_test.save()
        color_question, created = Question.objects.get_or_create(test_id= this_test, topic= 'Color')  
        color_question.save()
        color_answer, created = Answer.objects.get_or_create(description= colorquestion, question=color_question)
        color_answer.save()
        #MultipleChoice=multipleChoiceModel(colorquestion=colorquestion)
        #MultipleChoice.save()
        request.session['selected_test_id'] = this_test.id
        return HttpResponseRedirect('survey2')
    except:
        logging.error("Log message goes here.")
def question2(request):
    return render(request, 'survey/survey2.html')
def question2_save(request): #primary key of the test view 
    if request.method!="POST":
        return HttpResponseRedirect(reverse("question2"))
    else:

        #main_category_object = request.POST.get("main_category_object")
        #SEARCH FOR THE OBJECT QUERY HERE
        rangeInput = request.POST.get("rangeInput")
        requested_objects = []
        requested_urls=[]
        objectquestion = request.POST['result_data'].split(',')
            #logging.error(wanted_queries)
        query = functools.reduce(operator.or_,[Q(object_name__contains=query_obj) for query_obj in objectquestion])
        results = ObjectOption.objects.filter(query)
        results_list = list(results)
        #logging.error(results)
        object_relations_list = list(ObjectRelation.objects.all())
        for i in range(0, len(results)):
            for a in object_relations_list:
                if results_list[i].object_name==a.Object_Name_1 and a.object_relations>0.45:
                    for x in list(ObjectOption.objects.all()):
                        if x.object_name==a.Object_Name_2 and x not in results_list:
                            results_list.append(x)
                            #logging.error(a.Object_Name_2)  
                elif  results_list[i].object_name==a.Object_Name_2 and a.object_relations>0.45:     
                    for c in list(ObjectOption.objects.all()):
                        if c.object_name==a.Object_Name_2 and c not in results_list:
                            results_list.append(c)
                            #logging.error(a.Object_Name_2)
        #logging.error(results_list)     
        for d in range(0, len(results_list)):     
            requested_objects.append(results_list[d].artwork_name)
            requested_urls.append(results_list[d].url)         
            #requested_objects.append(results[i].artwork_name)
            #requested_urls.append(results[i].url)
        
        survey_artworks_weights = request.session['survey_artworks_weights']
        requested_objects = np.array(requested_objects)
        requested_objects = np.unique(requested_objects)
        requested_objects = list(requested_objects)
        for i in range(len(results_list)):
            object_confidence  = results_list[i].confidence
            if str(results_list[i].artwork_name) in survey_artworks_weights:
                survey_artworks_weights[str(results_list[i].artwork_name)]['score_from_objects_coef_and_user_input'] = object_confidence * int(rangeInput)
                #add the weight 
            else:
                #add the key
                survey_artworks_weights[str(results_list[i].artwork_name)] = {}
                survey_artworks_weights[str(results_list[i].artwork_name)]['score_from_colors_coef_and_user_input'] = 0
                survey_artworks_weights[str(results_list[i].artwork_name)]['total_score'] = 0
                survey_artworks_weights[str(results_list[i].artwork_name)]['score_from_objects_coef_and_user_input'] =  object_confidence * int(rangeInput)
                survey_artworks_weights[str(results_list[i].artwork_name)]['chosen_artist'] = 0
                #survey_artworks_weights[str(requested_objects[i])] = int(rangeInput)
        request.session['survey_artworks_weights'] = survey_artworks_weights   
        request.session['object_query'] = requested_objects
        request.session['requested_urls'] = requested_urls     
        #logging.error(survey_artworks_weights)


        #result_test = ObjectOption.objects.filter(object_name__contains='vase')
        #result_test= list(result_test)
        #requested_artworks_per_query.append(wanted_queries) #wine glass
        #for i in range(0, len(result_test)):
        #    requested_artworks_per_query.append(result_test[i].artwork_name)
        #requested_objects.append(requested_artworks_per_query)
        #logging.error(results)

        #objectquestion = [s.replace("'", "") for s in objectquestion]
        #objectquestion=request.POST.getlist("result_data")   #objectquestion[] object_array
    try:
        pk = request.session.get('selected_test_id')
        this_test = Test.objects.get(pk=pk)
        object_question = Question(test_id= this_test, topic= 'Object')
        object_question.save()     
        object_answer, created = Answer.objects.get_or_create(description= objectquestion, question=object_question)
        object_answer.save()   
        #MultipleChoice=multipleChoiceModel(objectquestion=objectquestion)
        #MultipleChoice.save()
        return HttpResponseRedirect('survey3')
    except:
        logging.error("Log message goes here.")

# def question3(request):
#     return render(request, 'survey/survey3.html')
# def question3_save(request):
#     if request.method!="POST":
#         return HttpResponseRedirect(reverse("question3"))
#     else:
#         rangeInput = request.POST.get("rangeInput")
#         genrequestion=request.POST.get("genrequestion")
#         requested_genre = []
#             #logging.error(wanted_queries)
#         query = functools.reduce(operator.or_,[Q(art_style__contains=query_obj) for query_obj in genrequestion])
#         results = GenreOption.objects.filter(query)
#         results_list = list(results)
#         for i in range(0, len(results)):
#             requested_genre.append(results[i].artwork_name)
#         request.session['requested_genre'] = requested_genre
#         requested_genre = np.array(requested_genre)
#         requested_genre = np.unique(requested_genre)
#         requested_genre = list(requested_genre)
#         survey_artworks_weights = request.session['survey_artworks_weights']
#         for i in range(len(requested_genre)):
#             if str(requested_genre[i]) in survey_artworks_weights:
#                 survey_artworks_weights[str(requested_genre[i])]['score_from_genre_coef_and_user_input'] = 100 
#                 #add the weight 
#             else:
#                 #add the key
#                 #add the key
#                 survey_artworks_weights[str(requested_genre[i])] = {}
#                 survey_artworks_weights[str(requested_genre[i])]['score_from_objects_coef_and_user_input'] = 0
#                 survey_artworks_weights[str(requested_genre[i])]['score_from_genre_coef_and_user_input'] = 100 #* int(rangeInput) CHECKHERE
#                 survey_artworks_weights[str(requested_genre[i])]['total_score'] = 0
#                 survey_artworks_weights[str(requested_genre[i])]['score_from_colors_coef_and_user_input'] = 0
#                 survey_artworks_weights[str(requested_genre[i])]['chosen_artist'] = 0


#         request.session['survey_artworks_weights'] = survey_artworks_weights
#         #logging.error(survey_artworks_weights)


#         #logging.error(requested_genre)

#     try:    
#         pk = request.session.get('selected_test_id')
#         this_test = Test.objects.get(pk=pk)
#         genre_question = Question(test_id= this_test, topic= 'Genre')    
#         genre_question.save()
#         genre_answer, created = Answer.objects.get_or_create(description= genrequestion, question= genre_question)
#         genre_answer.save()
#         return HttpResponseRedirect("survey4")
#         #return HttpResponseRedirect("success")
#     except:  
#         logging.error("Log message goes here.")

def artist_question(request):
    artists = ArtworkInfo.objects.values_list('people').distinct()
    return render(request, 'survey/survey4.html', {'artists': artists})
def artist_question_save(request):
    artwork_name_list = []
    if request.method != "POST":
        return HttpResponseRedirect(reverse("artist_question"))
    else:
        artist_list = request.POST.getlist("artist")
        for artist in artist_list:
            artwork_names = list(ArtworkInfo.objects.filter(Q(people_english=artist)).values_list('object_number', flat=True))
            for artwork in artwork_names:
                artwork_name_list.append(artwork)
            #artwork_names.append(artwork_name)


        #artist = artist.decode('utf-8')
        survey_artworks_weights = request.session['survey_artworks_weights']
        for i in range(len(artwork_name_list)):
            if str(artwork_name_list[i]) in survey_artworks_weights:
                survey_artworks_weights[str(artwork_name_list[i])]['chosen_artist'] = 100 
            else:
                survey_artworks_weights[str(artwork_name_list[i])] = {}
                survey_artworks_weights[str(artwork_name_list[i])]['score_from_objects_coef_and_user_input'] = 0
                survey_artworks_weights[str(artwork_name_list[i])]['total_score'] = 0
                survey_artworks_weights[str(artwork_name_list[i])]['score_from_colors_coef_and_user_input'] = 0
                survey_artworks_weights[str(artwork_name_list[i])]['chosen_artist'] = 100


        request.session['survey_artworks_weights'] = survey_artworks_weights
        #logging.error(survey_artworks_weights)

        return HttpResponseRedirect("success")


def recursive_lookup(key, d):
    if key in d:
        return d[key]

    for k, v in d.items():
        if isinstance(v, dict):
            result = recursive_lookup(key, v)

            if result:
                return k, result
def success(request):
    pk = request.session.get('selected_test_id')
    this_test = Test.objects.get(pk=pk)  
    new_list=[]
    clean_urls = []
    common_artworks_urls = []

    survey_artworks_weights = request.session['survey_artworks_weights']
    #add_total_Score for each artwork
    score_list_totalscore = []
    score_list_objectscore = []
    score_list_colorscore = []
    score_list_artistscore = []
    for scores in survey_artworks_weights.values():
        logging.error(int(scores['chosen_artist']))
        logging.error(int(scores['score_from_objects_coef_and_user_input']))
        logging.error(int(scores['score_from_colors_coef_and_user_input']))

        scores['total_score'] = int(scores['score_from_objects_coef_and_user_input']) + int(scores['score_from_colors_coef_and_user_input']) + int(scores['chosen_artist'])
        score_list_totalscore.append(scores['total_score'])
        score_list_objectscore.append(scores['score_from_objects_coef_and_user_input'])
        score_list_colorscore.append(scores['score_from_colors_coef_and_user_input'])
        #score_list_artistscore.append(scores['chosen_artist'])
    artwork_list_from_dic = list(survey_artworks_weights.keys())
    score_list_colorscore = np.multiply(softmax(minmax_scale(score_list_colorscore)), 1000.0).tolist()
    score_list_objectscore = np.multiply(softmax(minmax_scale(score_list_objectscore)), 1000.0).tolist()
    score_list_totalscore = np.multiply(softmax(minmax_scale(score_list_totalscore)), 1000.0).tolist()
    #score_list_artistscore = np.multiply(softmax(minmax_scale(score_list_artistscore)), 1000.0).tolist()
    for i in range(0, len(score_list_totalscore)):
        survey_artworks_weights[artwork_list_from_dic[i]]['total_score'] = round(score_list_totalscore[i])
        survey_artworks_weights[artwork_list_from_dic[i]]['score_from_objects_coef_and_user_input'] = round(score_list_objectscore[i])
        survey_artworks_weights[artwork_list_from_dic[i]]['score_from_colors_coef_and_user_input'] = round(score_list_colorscore[i])
        #survey_artworks_weights[artwork_list_from_dic[i]]['chosen_artist'] = round(score_list_artistscore[i])

    #logging.error(survey_artworks_weights)
    survey_artworks_weights = {k:v for k, v in sorted(survey_artworks_weights.items(), key=lambda item: (item[1]['chosen_artist'], item[1]['total_score']), reverse=True)}

    #logging.error(survey_artworks_weights)

    #lists contain names of the selected fields for each question
    requested_objects = request.session['object_query']
    #requested_genre = request.session['requested_genre']
    requested_color = request.session['requested_color']
    requested_color_urls =  request.session['requested_color_urls']
    requested_urls=request.session['requested_urls']
    artwork_list_from_dic = list(survey_artworks_weights.keys())
    for i in range(0, len(artwork_list_from_dic)):
        file_s = ObjectOption.objects.filter(artwork_name=artwork_list_from_dic[i]).values_list('url', flat=True).first()
        common_artworks_urls.append(file_s)

    for i in range(len(common_artworks_urls)):
        if(common_artworks_urls[i] != '#N/A' and common_artworks_urls[i] != None):
            clean_urls.append(common_artworks_urls[i])


    ##BURAYA ARTIST NAMEI EKLE-RESULTSA
    if(len(clean_urls) > 5):
        for i in range(0,5):
            found_artwork = ObjectOption.objects.filter(url=clean_urls[i]).values_list('artwork_name', flat=True).first()
            ##found artworkun art infosunu al
            try:
                artwork_info_object = ArtworkInfo.objects.filter(object_number=found_artwork).first()
            except:
                artwork_info_object = None
            finally:
                recommendation_score = survey_artworks_weights[found_artwork]['total_score']
                objects_score = survey_artworks_weights[found_artwork]['score_from_objects_coef_and_user_input']
                colors_score = survey_artworks_weights[found_artwork]['score_from_colors_coef_and_user_input']
                #key, value = recursive_lookup(found_artwork, survey_artworks_weights)
                #recommendation_score = str(survey_artworks_weights[j]['total_score'])
                #logging.error(recommendation_score)
                result, created = Result.objects.get_or_create(survey=this_test, found_url=clean_urls[i], recommendation_score= recommendation_score, color_score=colors_score, object_score= objects_score, artwork_information= artwork_info_object)
    else:
        for i in range(len(clean_urls)):
            found_artwork = ObjectOption.objects.filter(url=clean_urls[i]).values_list('artwork_name', flat=True).first()
            try:
                artwork_info_object = ArtworkInfo.objects.filter(object_number=found_artwork).first()
            except:
                artwork_info_object = None
            finally:
                recommendation_score = survey_artworks_weights[found_artwork]['total_score']
                objects_score = survey_artworks_weights[found_artwork]['score_from_objects_coef_and_user_input']
                colors_score = survey_artworks_weights[found_artwork]['score_from_colors_coef_and_user_input']
                result, created = Result.objects.get_or_create(survey=this_test, found_url=clean_urls[i], recommendation_score= recommendation_score, color_score=colors_score, object_score= objects_score, artwork_information= artwork_info_object) 
    wanted_results = Result.objects.filter(survey=this_test)
    #logging.error(wanted_results)
    #logging.error(pk)

    # for i in range(len(requested_genre)):
    #     for l in range(len(requested_objects)):
    #         for k in range(len(requested_color)):
    #             controller=True
    #             if requested_genre[i]==requested_objects[l]==requested_color[k]:
    #                 if(requested_urls[l]!='#N/A'):
    #                     logging.error(requested_genre[i])
    #                     for x in range(len(new_list)):
    #                         if(new_list[x]==requested_urls[l]):
    #                             controller=False
    #                     if(controller==True):
    #                         new_list.append(requested_urls[l])
                     


    #logging.error(new_list)
    #simdilik new list yerine clean_urls yazdim
    return render(request, 'survey/success.html',{"Result": wanted_results}) #{"URLS":new_list})

def answers_upload_objects(request):
    template = "answer_upload.html"
    data = ObjectOption.objects.all()

    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')

    #headers[i] = main_Category
    headers = set()
    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)
    next(io_string)
    csv_reader = csv.reader(io_string, delimiter=',', quotechar="|")
    headers.update(next(csv_reader, []))
    headers = list(headers)
    lenCol = len(next(csv_reader))

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        column = [i for i in column if i]
        column[2] = column[2].replace("%","")
        column[2] = float(column[2])
        _, created = ObjectOption.objects.update_or_create(artwork_name=column[0], object_name=column[1], 
            confidence= column[2], url=column[3])
    context = {}
    return render(request, template, context)

def answers_upload_genre(request):
    template = "answer_upload.html"
    data = GenreOption.objects.all()

    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')

    #headers[i] = main_Category
    headers = set()
    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)
    next(io_string)
    csv_reader = csv.reader(io_string, delimiter=',', quotechar="|")
    headers.update(next(csv_reader, []))
    headers = list(headers)
    lenCol = len(next(csv_reader))

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        column = [i for i in column if i]
        _, created = GenreOption.objects.update_or_create(artwork_name=column[0], art_style=column[1])
    context = {}
    return render(request, template, context)
def answers_upload_object_relations(request):
    template = "answer_upload.html"
    data = ObjectRelation.objects.all()

    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')

    #headers[i] = main_Category
    headers = set()
    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)
    next(io_string)
    csv_reader = csv.reader(io_string, delimiter=',', quotechar="|")
    headers.update(next(csv_reader, []))
    headers = list(headers)
    lenCol = len(next(csv_reader))

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        column = [i for i in column if i]
        _, created = ObjectRelation.objects.update_or_create(Object_Name_1=column[0], Object_Name_2=column[1], object_relations=column[2])
    context = {}
    return render(request, template, context)

def answers_upload_colors(request):
    template = "answer_upload.html"
    data = ColorOption.objects.all()

    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')

    #headers[i] = main_Category
    headers = set()
    dataset = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(dataset)
    next(io_string)
    csv_reader = csv.reader(io_string, delimiter=',', quotechar="|")
    headers.update(next(csv_reader, []))
    headers = list(headers)
    lenCol = len(next(csv_reader))

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        column = [i for i in column if i]
        column[8] = column[8].replace(';', ',')
        column[9] = column[9].replace(';', ',')
        column[10] = column[10].replace(';', ',')
        _, created = ColorOption.objects.update_or_create(artwork_name=column[1], color_one=column[2], color_two=column[3], color_three=column[4], 
            color_one_coefficient= column[5], color_two_coefficient= column[6], color_three_coefficient= column[7], colors1_rgb_list=column[8],
            colors2_rgb_list= column[9], colors3_rgb_list=column[10], url=column[11])
    context = {}
    return render(request, template, context)


def answers_upload_artwork_info(request):
    template = "answer_upload.html"
    #data = ColorOption.objects.all()

    if request.method == "GET":
        return render(request, template)
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'THIS IS NOT A CSV FILE')

    #headers[i] = main_Category
    headers = set()
    dataset = csv_file.read().decode('utf-8-sig')
    io_string = io.StringIO(dataset)
    next(io_string)
    csv_reader = csv.reader(io_string, delimiter=',', quotechar="|")
    headers.update(next(csv_reader, []))
    headers = list(headers)
    lenCol = len(next(csv_reader))

    for column in csv.reader(io_string, delimiter=',', quotechar="|"):
        column = [i for i in column if i]
        _, created = ArtworkInfo.objects.update_or_create(source_id=column[1], object_number=column[2], medium=column[3], on_view=column[4], 
            media= column[5], classification= column[6], people= column[7], primary_media=column[8],
            display_date= column[9], exhibitions=column[10], id_2=column[11], dimensions=column[12], people_english=column[13])

    context = {}
    return render(request, template, context)
# class SurveyStepsFormSubmission(SessionWizardView): # object kismini da koy
# 	#template_name = 'survey/survey.html'
# 	def get_template_names(self):
# 		return 'survey/survey_eyl.html'
# 	form_list = [MultipleChoiceGenreForm, MultipleChoiceObjectForm]
# 	def done(self, form_list, **kwargs):
# 		form_data = [form.cleaned_data for form in form_list]
# 		this_test = Test()
# 		this_test.save()
# 		#if form is valid
# 		selected_genres = form_list[0]
# 		this_question, created = GenreTestNew.objects.get_or_create(test_id=this_test)
# 		this_question.save()
# 		for selected_genre in selected_genres:
# 			one_answer, created = Answer.objects.get_or_create(answer= selected_genre, genre_test=this_question)
# 			one_answer.save()
# 		return render(self.request, 'survey/home_eyl.html', {
#     		'data': form_data
#     		})	

# FORMS = [("genre", MultipleChoiceGenreForm),
# ("object", MultipleChoiceObjectForm)]

# TEMPLATES = {"genre": "survey/survey3.html",
# "object": "survey/survey2.html"}

# class SurveyStepsFormSubmissionEylul(SessionWizardView):
#     def get_template_names(self):
#         return [TEMPLATES[self.steps.current]]
#     form_list = [MultipleChoiceGenreForm, MultipleChoiceObjectForm]
#     def done(self, form_list, **kwargs):
#         form_data = [form.cleaned_data for form in form_list]

#django form wizard terkedilebilir
#resimler renderlanmiyor
# class SurveyStepsFormSubmissionDoruk(SessionWizardView): # object kismini da koy
#     #template_name = 'survey/survey.html'
#     #create genre choices first

#     #add-> genretestnew.genre_choices.add()
#     def get_template_names(self):
#         return 'survey/survey_new.html'
#     form_list = [MultipleChoiceGenreForm, MultipleChoiceObjectForm, ColorForm]
#     def done(self, form_list, **kwargs):
#         form_data = [form.cleaned_data for form in form_list]
#         this_test = Test()
#         this_test.save()
#         #if form is valid
#         selected_genres = form_data[0] #buraya bak
#         selected_objects = form_data[1]
#         selected_color = form_data[2]
#         selected_objects = json.dumps([c.description for c in selected_objects['object_choices']])
#         selected_genres = json.dumps([c.description for c in selected_genres['genre_choices']])
#         #selected_color= json.dumps(selected_color)
#         if selected_genres:
#             try:
#                 selected_genres = json.loads(selected_genres)
#             except JSONDecodeError:
#                 pass
#         if selected_objects:
#             try:
#                 selected_objects = json.loads(selected_objects)
#             except JSONDecodeError:
#                 pass    
#         # if selected_color:
#         #     try:
#         #         selected_color = json.loads(selected_color)
#         #     except JSONDecodeError:
#         #         pass
#         genre_question, created = GenreTestNew.objects.get_or_create(test_id=this_test)
#         object_question, created = ObjectRecognitionTestNew.objects.get_or_create(test_id= this_test)
#         #color_question, created = ColorTest.get_or_create(test_id=this_test, color_answer= selected_color)
#         #color_question.save()
#         genre_question.save()
#         object_question.save()
#         for selected_genre in selected_genres:
#             #check the genre answer
#             genre_answer, created = GenreAnswer.objects.get_or_create(answer= selected_genre, genre_test=genre_question)
#             genre_answer.save()
#         for selected_object in selected_objects:
#             object_answer, created = ObjectAnswer.objects.get_or_create(answer= selected_object, object_test=object_question)
#             object_answer.save()
#         #color_answer,created = ColorTest.get()
#         return render(self.request, 'survey/survey_header.html', {
#             'data': form_data
#             })  