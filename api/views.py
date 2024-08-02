from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from openai import OpenAI
from decouple import config
from django.views.decorators.csrf import csrf_exempt
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests


store = {}
configuration = {"configurable":{"session_id":"testing_session"}}

SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o", api_key=SECRET_KEY)
client = ChatOpenAI(api_key=SECRET_KEY)
parser = StrOutputParser()
trimmer=trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=True,
    start_on="system",
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def bing_search(query):
    bing_key=config('BING_API_KEY')
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML","mkt":"en-US","count":5}
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    print(search_results)
    return search_results

@csrf_exempt
def chat_with_openai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            user_input = data.get('message')
            if user_input:
               
                system_template = "You are a helpful assistant. You are assisting a user"

                prompt_template= ChatPromptTemplate.from_messages([
                    ("system", system_template),
                    ("user", "{text}"),
                ])

                chain = prompt_template | client | parser 
                with_history = RunnableWithMessageHistory(chain,get_session_history,config=configuration)
                response = with_history.invoke({"text": user_input})
                # result = parser.invoke(response)
                # bingSearch = bing_search(user_input)
                # print(bingSearch)
                return JsonResponse({"response": response})
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
                print(messages)
                chain = client | parser  # Define the message chain
                # Assuming `client` is already defined and initialized with your Langchain credentials
                response = chain.invoke(messages)
                
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

            hardcoded_password = "demo_password"
            if username == "demo_user" and password == hardcoded_password:
                return JsonResponse({"message": "Login successful"})
            else:
                return JsonResponse({"error": "Invalid credentials"}, status=401)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Request must be POST"}, status=405)