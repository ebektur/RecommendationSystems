# Generated by Django 2.2.6 on 2022-03-06 14:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('predictors', '0012_auto_20220306_1454'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='colortest',
            name='color_answer',
        ),
        migrations.AlterField(
            model_name='colortest',
            name='test_id',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='color_question', to='predictors.Test'),
        ),
        migrations.AlterField(
            model_name='objectanswer',
            name='object_test',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='object_answers', to='predictors.ObjectRecognitionTestNew'),
        ),
        migrations.AlterField(
            model_name='objectrecognitiontestnew',
            name='test_id',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='object_question', to='predictors.Test'),
        ),
        migrations.CreateModel(
            name='ColorAnswer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('answer', models.CharField(db_index=True, max_length=240)),
                ('color_test', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='color_answers', to='predictors.ColorTest')),
            ],
        ),
    ]