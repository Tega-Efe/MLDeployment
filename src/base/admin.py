from django.contrib import admin

# Register your models here.
from . import models


admin.site.register(models.CustomUser)
admin.site.register(models.Prediction)