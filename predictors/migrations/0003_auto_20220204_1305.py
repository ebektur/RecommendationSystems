# Generated by Django 2.2.6 on 2022-02-04 13:05

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0002_auto_20220203_1359'),
    ]

    operations = [
        migrations.CreateModel(
            name='GenreAnswer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('answer', models.CharField(db_index=True, max_length=240)),
                ('genre_test', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='predictors.GenreTestNew')),
            ],
        ),
        migrations.CreateModel(
            name='ObjectAnswer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('answer', models.CharField(db_index=True, max_length=240)),
                ('object_test', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='predictors.ObjectRecognitionTestNew')),
            ],
        ),
        migrations.DeleteModel(
            name='Answer',
        ),
    ]