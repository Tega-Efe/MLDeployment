# Create your views here.
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy
from django.contrib.auth import login, logout
from django.views.generic.edit import FormView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import CustomUserCreationForm, CustomAuthenticationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.list import ListView

from django.core.files.storage import default_storage
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd
from django.views import View
from .models import Prediction, CustomUser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs

class CustomLoginView(FormView):
    template_name = 'base/login.html'
    form_class = CustomAuthenticationForm

    def form_valid(self, form):
        user = form.get_user()

        if user is not None:
            login(self.request, user)
            self.request.session.save()

            # Debug: Check session ID generation
            session_id = self.request.session.session_key
            print(f"Session ID after login: {session_id}")
            print(f"Is authenticated: {self.request.user.is_authenticated}")

            return redirect('dashboard')
        else:
            print("User authentication failed in form_valid()")

        return super().form_invalid(form)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)

    def get_success_url(self):
        return reverse_lazy('dashboard')


class Dashboard(LoginRequiredMixin,ListView):
    model = Prediction
    template_name = 'base/dashboard.html'
    context_object_name = 'predictions'


    def get_queryset(self):
        if not self.request.user.is_authenticated:
            return Prediction.objects.none()
        return Prediction.objects.filter(user=self.request.user).order_by('-timestamp')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user

        # Debug Information
        print(f"User authenticated: {user.is_authenticated}")
        print(f"Username: {user.username}")
        print(f"User ID: {user.id}")
        print(f"User type: {type(user)}")

        if user.is_authenticated:
            # Ensure we are using the custom user instance
            if isinstance(user, CustomUser):
                context['user'] = user
                context['prediction_count'] = Prediction.objects.filter(user=user).count()
            else:
                print("request.user is not an instance of CustomUser")
                context['prediction_count'] = 0
        else:
            context['prediction_count'] = 0

        return context

def delete_prediction(request, prediction_id):
    if request.method == "POST":
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        prediction.delete()
        return JsonResponse({"status": "success", "message": "Prediction deleted successfully"})
    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)



class RegisterPage(FormView):
    template_name = 'base/signup.html'
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('dashboard')

    def form_valid(self, form):
        user = form.save()
        user.save()  # Ensure the user is saved

        login(self.request, user)

        return redirect(self.success_url)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)

    def get(self, *args, **kwargs):
        if self.request.user.is_authenticated:
            return redirect('dashboard')
        return super(RegisterPage, self).get(*args, **kwargs)

def about(request):
    return render(request, 'base/about.html')

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('login')  

    # GET requests also redirect to the login page
    elif request.method == 'GET':
        logout(request)
        return redirect('login')  

    # If the request method is neither GET nor POST, return an error response
    return JsonResponse({'status': 'failed'}, status=400)


from django.http import JsonResponse

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../savedModels/imageclassifier__3.h5')
model = load_model(MODEL_PATH)
try:
    # Load labels from Excel
    labels_df = pd.read_excel("labels.xlsx")  # Ensure correct file path

    # Debug: Print first few rows
    print("First few rows of labels_df:")
    print(labels_df.head())

    # Debug: Check for NaN values
    print("\nChecking for NaN values:")
    print(labels_df.isna().sum())

    # Debug: Check for duplicate ClassId values
    print("\nChecking for duplicate ClassId values:")
    print(labels_df["ClassId"].duplicated().sum())

    # Drop NaN rows and ensure correct types
    labels_df = labels_df.dropna().reset_index(drop=True)
    labels_df["ClassId"] = labels_df["ClassId"].astype(int)  # Ensure integer index

    # Create mapping dictionary
    label_mapping = dict(zip(labels_df["ClassId"], labels_df["Name"]))

    # Debug: Print final mapping
    print("\nFinal label mapping:")
    print(label_mapping)

except Exception as e:
    print(f"Error loading labels.xlsx: {e}")
    label_mapping = {}

class PredictImageView(View):
    template_name = 'base/upload.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def post(self, request):
        if 'image' not in request.FILES:
            print("No image uploaded.")  # Debugging output
            return JsonResponse({'error': 'No image uploaded'}, status=400)
        
        image_file = request.FILES['image']
        print(f"Received image: {image_file.name}")  # Debugging output
        
        image_path = default_storage.save('uploads/' + image_file.name, image_file)
        full_image_path = default_storage.path(image_path)
        print(f"Saved image path: {full_image_path}")  # Debugging output

        class_name, confidence = self.process_and_predict(full_image_path)
        
        user_instance = None
        if request.user.is_authenticated:
            try:
                user_instance = CustomUser.objects.get(pk=request.user.pk)
            except CustomUser.DoesNotExist:
                print("Authenticated user not found in database.")  # Debugging output
                user_instance = None
        
        prediction_entry = Prediction.objects.create(
            user=user_instance,
            image=image_path,
            predicted_class=class_name,
            confidence=float(confidence)
        )
        print(f"Prediction saved to database: {class_name} ({confidence:.2f}%)")  # Debugging output
        
        return JsonResponse({
            'image_url': prediction_entry.image.url,
            'class_name': class_name,
            'confidence': float(round(confidence, 2))
        })
    
    def process_and_predict(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Image could not be read. Check file format and path.")
            print(f"\n=== Testing image: {image_path} ===")  # Debugging output
            print(f"Original Image Shape: {img.shape}")  # Debugging output
            
            resize = tf.image.resize(img, (128, 128))
            grayscale = cv2.cvtColor(resize.numpy(), cv2.COLOR_BGR2GRAY)
            scaled = ((grayscale / 255.0) * 70) + 13
            print(f"Resized Image Shape: {resize.shape}")  # Debugging output
            

            print(f"Grayscale Image Shape: {grayscale.shape}")  # Debugging output
            
            scaled = ((grayscale / 255.0) * 70) + 13
            print(f"Scaled Image - Min: {scaled.min()}, Max: {scaled.max()}")  # Debugging output
            
            processed = np.expand_dims(scaled, axis=-1)
            processed = np.expand_dims(processed, axis=0)
            print(f"Processed Image Shape: {processed.shape}")  # Debugging output
            
            predictions = model.predict(processed, verbose=0)
            print(f"Raw Predictions: {predictions}")  # Debugging output
            
            predicted_index = np.argmax(predictions)
            print(f"Predicted Index: {predicted_index}")  # Debugging output
            
            class_name = label_mapping[predicted_index]
            confidence = predictions[0][predicted_index] * 100
            
            print(f"Final Prediction: {class_name} ({confidence:.2f}%)")  # Debugging output
            print(f"\nPredicted Class: {class_name} ({predictions[0][predicted_index] * 100:.2f}%)")


            return class_name, confidence
        except Exception as e:
            print(f"Error in process_and_predict: {e}")
            return "Error", 0.0






