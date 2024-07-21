from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from openai import OpenAI
from decouple import config
from django.views.decorators.csrf import csrf_exempt
import json
import os

SECRET_KEY = config('SECRET_KEY')
client = OpenAI(api_key=SECRET_KEY)
@csrf_exempt
def chat_with_openai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            user_input = data.get('message')
            if user_input:
                response = client.chat.completions.create(model="gpt-4",   # Specify the model you are using
                messages=[
                    {"role": "system", "content": "You are connected to a help assistant."},
                    {"role": "user", "content": user_input},
                ])
                return JsonResponse({"response": response.choices[0].message.content})
            else:
                return JsonResponse({"error": "No message provided"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Request must be POST"}, status=405)

@csrf_exempt
def simulate_conversation(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            user_input = data.get('message')
            previous_message = data.get('previousMessage')  # Get the previous message
            topic = data.get('topic')
            if user_input:
                # Create message history for the model
                message_history = [
                    {"role": "system", "content": "You are conversing about the topic: " + topic},
                ]
                if previous_message:
                    message_history.append({"role": "user", "content": previous_message})
                message_history.append({"role": "user", "content": user_input})

                response = client.chat.completions.create(
                    model=data.get('model'),
                    max_tokens=data.get('maxTokens'),
                    messages=message_history
                )
                
                return JsonResponse({"response": response.choices[0].message.content})
            else:
                return JsonResponse({"error": "No message provided"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Request must be POST"}, status=405)

@csrf_exempt
def user_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            username = data.get('username')
            password = data.get('password')

            hardcoded_password = "demo_password_kangacook"
            if username == "roulettech_demo_user" and password == hardcoded_password:
                return JsonResponse({"message": "Login successful"})
            else:
                return JsonResponse({"error": "Invalid credentials"}, status=401)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Request must be POST"}, status=405)