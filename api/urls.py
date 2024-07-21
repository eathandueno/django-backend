from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.user_login, name='login'),
    path('chat/', views.chat_with_openai, name='chat'),
    path('simulate/', views.simulate_conversation, name='simulate'),
]