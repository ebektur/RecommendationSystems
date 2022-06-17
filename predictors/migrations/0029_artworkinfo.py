# Generated by Django 2.2.6 on 2022-06-03 13:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0028_auto_20220428_1547'),
    ]

    operations = [
        migrations.CreateModel(
            name='ArtworkInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_id', models.CharField(max_length=300)),
                ('object_number', models.CharField(max_length=300)),
                ('medium', models.CharField(max_length=300)),
                ('on_view', models.CharField(max_length=300)),
                ('media', models.CharField(max_length=300)),
                ('classification', models.CharField(max_length=300)),
                ('people', models.CharField(max_length=300)),
                ('primary_media', models.CharField(max_length=300)),
                ('display_date', models.CharField(max_length=300)),
                ('exhibitions', models.CharField(max_length=300)),
                ('id_2', models.CharField(max_length=300)),
                ('dimensions', models.CharField(max_length=300)),
            ],
        ),
    ]