# Generated by Django 2.2.6 on 2022-03-07 14:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0017_auto_20220307_1424'),
    ]

    operations = [
        migrations.AlterField(
            model_name='question',
            name='topic',
            field=models.CharField(max_length=30),
        ),
    ]
