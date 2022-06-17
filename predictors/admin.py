from django.contrib import admin
from .models import *
#from merged_inlines.admin import MergedInlineAdmin

admin.site.register(Test)


# class MainColorInline(admin.TabularInline):
# 	model = ColorChoice

# class ColorChoiceAdmin(admin.ModelAdmin):
# 	inlines = [MainColorInline]

# admin.site.register(MainColor, ColorChoiceAdmin)


##SILINMESIN SIMDILIK

# class QuestionInline(admin.TabularInline):
# 	model = Question

# class QuestionTopicAdmin(admin.ModelAdmin):
# 	inlines = [QuestionInline]

# admin.site.register(QuestionTopic, QuestionTopicAdmin)

class AnswerInline(admin.TabularInline):
	model = Answer
class QuestionAdmin(admin.ModelAdmin):
	inlines = [AnswerInline]

admin.site.register(Question, QuestionAdmin)
admin.site.register(Answer)
admin.site.register(ObjectOption)
admin.site.register(GenreOption)
admin.site.register(ColorOption)
admin.site.register(ObjectRelation)
admin.site.register(ArtworkInfo)
##SILINEBILIR
# class GenreChoiceInline(admin.TabularInline):
# 	model = GenreChoice

# class GenreChoiceAdmin(admin.ModelAdmin):
# 	inlines = [GenreChoiceInline]

# admin.site.register(GenreQuestion, GenreChoiceAdmin)

# class ObjectChoiceInline(admin.TabularInline):
# 	model = ObjectChoice

# class ObjectChoiceAdmin(admin.ModelAdmin):
# 	inlines = [ObjectChoiceInline]

# admin.site.register(ObjectQuestion, ObjectChoiceAdmin)

# class ColorChoiceInline(admin.TabularInline):
# 	model = ColorChoice

# class ColorChoiceAdmin(admin.ModelAdmin):
# 	inlines = [ColorChoiceInline]

# admin.site.register(ColorQuestion, ColorChoiceAdmin)


admin.site.register(Result)

