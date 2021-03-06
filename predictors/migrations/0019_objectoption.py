# Generated by Django 2.2.6 on 2022-03-22 20:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0018_auto_20220307_1428'),
    ]

    operations = [
        migrations.CreateModel(
            name='ObjectOption',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('confidence', models.FloatField()),
                ('url', models.CharField(max_length=300)),
                ('object_name', models.CharField(max_length=300)),
                ('artwork_name', models.CharField(max_length=300)),
            ],
        ),
    ]
