from django.forms import ModelForm
from django.forms import Form
from django.forms import MultipleChoiceField
from django.forms import ModelChoiceField
from django.forms import ModelMultipleChoiceField
from django.forms import CheckboxSelectMultiple
from django.forms import RadioSelect
#from colorfield.widgets import ColorWidget
from .models import *
from django import forms
from django.utils.safestring import mark_safe
from string import Template
from django.forms import ImageField

# class CustomChoiceField(ModelChoiceField):
# 	def label_from_instance(self, obj):
# 		return mark_safe("<img src='%s'/>" % obj.color_image.url)

# class ImagePreviewWidget(forms.widgets.FileInput):
#     def render(self, name, value, attrs=None, **kwargs):
#         html =  Template("""<img src="$link"/>""")
#         return mark_safe(html.substitute(link=value))

# class MultipleChoiceGenreForm(ModelForm):
# 	genre_choices = ModelMultipleChoiceField(queryset=GenreChoice.objects.all(), widget=CheckboxSelectMultiple())
# 	class Meta:
# 		model = GenreQuestion
# 		exclude = ['test_id']

# class MultipleChoiceObjectForm(ModelForm):
# 	object_choices = ModelMultipleChoiceField(queryset=ObjectChoice.objects.all(), widget=CheckboxSelectMultiple())
# 	class Meta:
# 		model = ObjectQuestion
# 		exclude = ['test_id']

# class ColorForm(ModelForm):
# 	color_choices = CustomChoiceField(queryset=ColorChoice.objects.all(), widget= RadioSelect) # widget= ImagePreviewWidget, widget= ColorWidget

# 	#color_choices = ImageField(widget=ImagePreviewWidget)
# 	class Meta:
# 		model = ColorQuestion
# 		exclude = ['test_id', 'color_answer']
	# def __init__(self, *args, **kwargs):
	# 	super(ColorForm, self).__init__(data=data, files=files, *args, **kwargs)
# class ItemForm(forms.Form): object

#     # here we use a dummy `queryset`, because ModelChoiceField
#     # requires some queryset
#     item_field = forms.ModelChoiceField(queryset=Item.objects.none())

#     def __init__(self, item_id):
#         super(ItemForm, self).__init__()
#         self.fields['item_field'].queryset = Item.objects.filter(id=item_id)