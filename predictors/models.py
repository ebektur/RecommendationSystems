from django.db import models

GENRES = [('Surrealism', 'Surrealism'),
('Romanticism', 'Romanticism'),
('Rococo', 'Rococo'),
('Pop Art', 'Pop Art'),
('Neoclassicism', 'Neoclassicism')]

OBJECTS = [('Vase', 'Vase'),
('Statue', 'Statue'),
('Decorative object', 'Decorative object'),
('Table', 'Table'),]

#ONE TEST WILL HAVE MANY MODEL PREDICTORS (QUESTIONS)
#ONE QUESTIONS MAY HAVE MANY ANSWERS 
class QuestionTopic(models.Model):
	TOPICS = (
		('Genre', 'Genre'),
		('Object', 'Object'),
		('Color', 'Color'),
		)
	selected_topic = models.CharField(max_length=30) #choices
	def __str__(self):
		return str(self.selected_topic)
class ObjectRelation(models.Model):
    Object_Name_1 = models.CharField(max_length=300)
    Object_Name_2 = models.CharField(max_length=300)
    object_relations = models.FloatField()
    def __str__(self):
        return str(self.id)
class MainColor(models.Model):
	MAIN_COLORS = (
		('R', 'Red'),
		('G', 'Green'),
		('B', 'Blue'),
		)
	dominant_color = models.CharField(max_length=30, choices=MAIN_COLORS)
	def __str__(self):
		return str(self.dominant_color)

class Test(models.Model): #QUIZ
	time_added = models.DateTimeField(auto_now_add = True)
	#topic,number of questions, time
	def __str__(self):
		return str(self.id)

	def get_questions(self): #get the question model
		return self.genre_question.all(), self.object_question.all(), self.color_question.all()

class Question(models.Model):
	test_id= models.ForeignKey(Test, on_delete= models.CASCADE, blank=True, related_name= 'question')
	topic = models.CharField(max_length=30)
	#topic = models.ForeignKey(QuestionTopic, on_delete=models.CASCADE, blank=True)
	def __str__(self):
		return str(self.id)
	def get_answer(self):
		return self.answer.all()

class Answer(models.Model):
	description = models.CharField(max_length=300)
	image = models.ImageField(upload_to= 'answers_images/', blank=True)
	question = models.ForeignKey(Question, on_delete=models.CASCADE, blank=True, related_name= 'answer')
	def __str__(self):
		return str(self.description)

class ObjectOption(models.Model):
	confidence = models.FloatField()
	url = models.CharField(max_length=300)
	object_name = models.CharField(max_length=300)
	artwork_name = models.CharField(max_length=300)
	def __str__(self):
		return str(self.id)

class GenreOption(models.Model):
	artwork_name = models.CharField(max_length=300)
	art_style = models.CharField(max_length=300)
	def __str__(self):
		return str(self.id)

class ColorOption(models.Model):
	artwork_name = models.CharField(max_length=300)
	color_one = models.CharField(max_length=300)
	color_two = models.CharField(max_length=300)
	color_three = models.CharField(max_length=300)
	color_one_coefficient = models.FloatField()
	color_two_coefficient = models.FloatField()
	color_three_coefficient = models.FloatField()
	colors1_rgb_list = models.CharField(max_length=300)
	colors2_rgb_list = models.CharField(max_length=300)
	colors3_rgb_list = models.CharField(max_length=300)
	url = models.CharField(max_length=300)

	def __str__(self):
		return str(self.id)

class ArtworkInfo(models.Model):
	source_id = models.CharField(max_length=300)
	object_number = models.CharField(max_length=300)
	medium = models.CharField(max_length=300)
	on_view = models.CharField(max_length=300)
	media = models.CharField(max_length=300)
	classification = models.CharField(max_length=300)
	people = models.CharField(max_length=300)
	primary_media = models.CharField(max_length=300)
	display_date = models.CharField(max_length=300)
	exhibitions = models.CharField(max_length=300)
	id_2 = models.CharField(max_length=300)
	dimensions = models.CharField(max_length=300)
	people_english = models.CharField(max_length=300)

# class GenreQuestion(models.Model): #QUESTION
# 	test_id = models.ForeignKey(Test, on_delete=models.CASCADE, blank=True, related_name= 'genre_question')
# 	#genre_choices = models.ManyToManyField(GenreChoice) #text
# 	#TotalGenreChoices = models.CharField(max_length=50, choices=GENRES, default=0)
# 	#number of question
# 	def __str__(self):
# 		return str(self.id)
# 	def get_answer_genre(self):
# 		return self.genre_answers.all()

# class ObjectQuestion(models.Model):  
# 	test_id = models.ForeignKey(Test, on_delete=models.CASCADE, blank=True, related_name='object_question')
# 	#object_choices= models.ManyToManyField(ObjectChoice) #tags
# 	def __str__(self):
# 		return str(self.id)
# 	def get_answer_object(self):
# 		return self.object_answers.all()

# class ColorQuestion(models.Model):
# 	test_id = models.ForeignKey(Test, on_delete= models.CASCADE, blank=True, related_name='color_question')
# 	#color_choices= models.ManyToManyField(ColorChoice)
# 	#color_answer = models.CharField(max_length=240, db_index=True) #one answer only question so no coloranswer
# 	def __str__(self):
# 		return str(self.id)
# 	def get_answer_color(self):
# 		return self.color_answers.all()

# class ObjectChoice(models.Model):
# 	description = models.CharField(max_length=300)
# 	object_question = models.ForeignKey(ObjectQuestion, on_delete=models.CASCADE, blank=True, related_name='object_answers')
# 	def __str__(self):
# 		return str(self.description)

# class GenreChoice(models.Model): #tag
# 	description = models.CharField(max_length=300)
# 	genre_question = models.ForeignKey(GenreQuestion, on_delete=models.CASCADE, blank=True, related_name='genre_answers')
# 	def __str__(self):
# 		return str(self.description)

# class MainColor(models.Model):
# 	MAIN_COLORS = (
# 		('R', 'Red'),
# 		('G', 'Green'),
# 		('B', 'Blue'),
# 		)
# 	dominant_color = models.CharField(max_length=30, choices=MAIN_COLORS)
# 	def __str__(self):
# 		return str(self.dominant_color)

# class ColorChoice(models.Model): #ANSWER
# 	color = models.CharField(max_length=300)
# 	color_image = models.ImageField(upload_to= 'colors_to_choose/')
# 	main_color = models.ForeignKey(MainColor, on_delete=models.CASCADE, blank=True)
# 	color_question = models.ForeignKey(ColorQuestion, on_delete=models.CASCADE, blank=True, related_name= 'color_answers')
	#def __str__(self):
	#	return str(self.color)

class Result(models.Model):
	survey = models.ForeignKey(Test, on_delete= models.CASCADE, blank=True)
	found_url = models.CharField(max_length=300)
	recommendation_score = models.CharField(max_length=300)
	color_score = models.CharField(max_length=300)
	object_score = models.CharField(max_length=300)
	artwork_information = models.ForeignKey(ArtworkInfo,
		null= True,
		blank= True,
		on_delete= models.SET_NULL)

	def result_urls_as_list(self):
		return self.found_url.split('\n')
	def __str__(self):
		return str(self.pk)	


##can be deleted soon
# class GenreAnswer(models.Model): #RESULTS
# 	answer = models.CharField(max_length=240, db_index=True) #text
# 	genre_test = models.ForeignKey(GenreTestNew, on_delete= models.CASCADE, blank=True, related_name= 'genre_answers')
# 	def __str__(self):
# 		return str(self.answer) 
# 	#object_test = models.ManyToManyField(ObjectRecognitionTestNew) #foreign key before

# class ObjectAnswer(models.Model): #results
# 	answer = models.CharField(max_length=240, db_index=True) 
# 	object_test= models.ForeignKey(ObjectRecognitionTestNew, on_delete= models.CASCADE, blank=True, related_name='object_answers')
# 	def __str__(self):
# 		return str(self.answer)

# class ColorAnswer(models.Model):
# 	answer= models.CharField(max_length=240, db_index=True)
# 	color_test= models.ForeignKey(ColorTest, on_delete= models.CASCADE, blank=True, related_name='color_answers')
# 	def __str__(self):
# 		return str(self.answer)

#CLASS RESULT?

class multipleChoiceModel(models.Model):
	colorquestion=models.CharField(max_length=240, db_index=True)
	objectquestion=models.CharField(max_length=240, db_index=True)
	genrequestion=models.CharField(max_length=240, db_index=True)
	def __str__(self):
	    return str(self.id)