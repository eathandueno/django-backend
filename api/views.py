from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from openai import OpenAI
from decouple import config
from django.views.decorators.csrf import csrf_exempt
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4", api_key=SECRET_KEY)
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
                messages = [
                    SystemMessage(content="You are conversing about the topic: " + topic),
                ]
                if previous_message:
                    messages.append(HumanMessage(content=previous_message))
                messages.append(HumanMessage(content=user_input))

                # Assuming `client` is already defined and initialized with your Langchain credentials
                response = client.invoke(messages)

                # The response handling here might need adjustment based on the actual response structure from Langchain
                return JsonResponse({"response": response})
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