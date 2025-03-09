from django.urls import path
from . import views

urlpatterns = [
    path('', views.Dashboard.as_view(), name='dashboard'), 
    path('dashboard/', views.Dashboard.as_view(), name='dashboard'), 
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('register/', views.RegisterPage.as_view(), name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.PredictImageView.as_view(), name='predictor'),
    path('predict/', views.PredictImageView.as_view(), name='predict'),
    path('delete-prediction/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    path('about/', views.about, name='about'),
]


