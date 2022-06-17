from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    #path('object_prediction', views.index, name='homepage'),
    path('admin/', admin.site.urls),
    path('genre/', views.genre, name='genre'),
    path('', include('predictors.urls')),#monitor app url
    ]
