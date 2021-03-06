# Generated by Django 2.2.6 on 2022-03-06 14:54

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0011_colortest_color_answer'),
    ]

    operations = [
        migrations.AlterField(
            model_name='genreanswer',
            name='genre_test',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='genre_answers', to='predictors.GenreTestNew'),
        ),
        migrations.AlterField(
            model_name='genretestnew',
            name='test_id',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='genre_question', to='predictors.Test'),
        ),
    ]
