from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError
from .models import CustomUser
import re


class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['email', 'firstname', 'lastname', 'mobile', 'password1', 'password2']

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if not password1 or not password2:
            raise ValidationError("Both password fields are required.")

        if password1 != password2:
            raise ValidationError("Passwords do not match.")

        if len(password1) < 8:
            raise ValidationError("Password must be at least 8 characters long.")

        if not re.search(r'[A-Z]', password1):
            raise ValidationError("Password must contain at least one uppercase letter.")

        if not re.search(r'\d', password1):
            raise ValidationError("Password must contain at least one digit.")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password1):
            raise ValidationError("Password must contain at least one special character.")

        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        if not user.username:  # Ensure username is set from email
            user.username = user.email.split('@')[0]
        if commit:
            user.save()
        return user



from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate

from django import forms

class CustomAuthenticationForm(AuthenticationForm):
    username = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))

    def clean(self):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if username and password:
            self.user_cache = authenticate(username=username, password=password)

            if self.user_cache is None:
                raise forms.ValidationError("Invalid username or password.")

        return self.cleaned_data