from django.urls import path
from . import views

urlpatterns = [
	#path('', views.survey, name = 'survey'),
    path('genre_preload', views.genre_preload2, name = 'genre_preload'),
    path('object_detection_preload', views.vgg_model_view, name = 'object_detection_preload'),
    path('genre_test_multiple', views.genre_test_multiple, name = 'genre_test_multiple'),
    path('success', views.success, name = 'success'),
    path('object_test_multiple', views.object_test_multiple, name = 'object_test_multiple'),
    path('upload-options-csv-objects', views.answers_upload_objects, name= "upload-options-csv-objects"),
    path('upload-options-csv-colors', views.answers_upload_colors, name= "upload-options-csv-colors"),
    path('upload-options-csv-genre', views.answers_upload_genre, name= "upload-options-csv-genre"),
    path('upload-options-csv-artwork-info', views.answers_upload_artwork_info, name= "upload-options-csv-artwork-info"),
    path('upload-options-csv-objectrelation', views.answers_upload_object_relations, name= "upload-options-csv-objectrelation"),

    #path('survey_submission', views.SurveyStepsFormSubmission.as_view(), name= 'survey_submission'),
    #path('survey_submission_new', views.SurveyStepsFormSubmissionDoruk.as_view(), name= 'survey_new'),
    path('survey',views.question, name= 'survey'),
    path('survey2',views.question2, name= 'survey2'),
    path('survey3',views.artist_question, name= 'survey3'),
    #path('survey4', views.artist_question, name='survey4'),
    path('survey_save',views.question_save, name= 'survey_save'),
    path('survey2_save',views.question2_save, name= 'survey2_save'),
    path('survey3_save',views.artist_question_save, name= 'artist_question_save'),
    #path('artist_question_save', views.artist_question_save, name='artist_question_save')    
    ]