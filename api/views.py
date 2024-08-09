from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from openai import OpenAI
from decouple import config
from django.views.decorators.csrf import csrf_exempt
import json
import re
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
import xml.etree.ElementTree as ET
from datetime import datetime

store = {}
configuration = {"configurable":{"session_id":"testing_session"}}
SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o", api_key=SECRET_KEY)
parser = StrOutputParser()
trimmer=trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=True,
    start_on="system",
)

template = """Answer the following questions as best you can. You have access to the following tools:

- Bing Search: Search Bing for information.
- Arxiv Search: Search Arxiv for papers.
- Get Time and Date: Retrieve the current date and time.

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [bing_search, arxiv_search, get_time_and_date]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {messages}

Thought:
"""




def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@tool
def bing_search(query, searchType = "WebPages", count = 1):
    """
    Search Bing and retrieve the top 3 results, presenting relevant snippets.

    Args:
        query (str): The text query for which search results are desired.
        searchType (str): The category of search results to be retrieved. This should be one of the following options:
            - "WebPages": Retrieves standard web page search results.
            - "Images": Retrieves image search results.
            - "News": Retrieves news articles related to the query.
            - "Videos": Retrieves video content related to the query.
            - "Computations": Retrieves results that involve calculations or computations.
            - "TimeZone": Retrieves information related to time zones.
            - "Places": Retrieves geographical location results.
            - "RelatedSearches": Retrieves queries related to the initial search term.

    Returns:
        str: A string of the cleaned snippets of the defined number of search results.
    """
    bing_key = config('BING_API_KEY')
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {
        "q": query,
        "textDecorations": True,
        "textFormat": "HTML",
        "mkt": "en-US",
        "count": count,
        "offset": 0,
        "responseFilter": searchType
    }
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    snippets = []
    if searchType.lower() == "webpages" and "webPages" in search_results:
        for result in search_results["webPages"]['value']:
            clean_snippet = remove_html_tags(result['snippet'])
            snippets.append(clean_snippet)
    elif searchType.lower() == "news" and "news" in search_results:
        for article in search_results["news"]['value']:
            clean_snippet = remove_html_tags(article['name'])
            snippets.append(clean_snippet)
    elif searchType.lower() == "images" and "images" in search_results:
        for image in search_results["images"]['value']:
            snippets.append(image['name'])
    elif searchType.lower() == "videos" and "videos" in search_results:
        for video in search_results["videos"]['value']:
            snippets.append(video['name'])
    elif searchType.lower() == "relatedsearches" and "relatedSearches" in search_results:
        for related_search in search_results["relatedSearches"]['value']:
            snippets.append(related_search['text'])
    else:
        snippets.append("No results found")

    return " \n ".join(snippets)


def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)  # This regex matches any text within <>, including the brackets
    return clean_text

@tool
def get_time_and_date():
    """
    Retrieve the current date and time.

    Returns:
        str: A string representing the current date and time in the format "YYYY-MM-DD HH:MM:SS".
    """
    # Get the current date and time
    current_datetime = datetime.now()
    
    # Format the date and time as a string in the format "YYYY-MM-DD HH:MM:SS"
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_datetime

@tool
def arxiv_search(search_query, max_results=1):
    """
    Search arXiv for papers matching the search query.

    Args:
        search_query: The search query to use.
        max_results: The maximum number of results to return.
    
    
    Returns:
        A list of strings, each containing the title, summary, publish_date and the authors.
    """


    
    # Construct the URL for the arXiv API request
    url = f'http://export.arxiv.org/api/query?search_query={search_query}&max_results={max_results}'

    # Send the GET request to the arXiv API
    response = requests.get(url)
    
    # Raise an exception if the request was unsuccessful
    response.raise_for_status()
    results=[]

    
    paper = {}
        # Extract the data from the XML entry
    paper = extract_data(response.text)
    
    results.append(paper)

    return results

def extract_data(xml):
    # Extract title
    title_start = xml.find('<title>') + 7
    title_end = xml.find('</title>', title_start)
    title = xml[title_start:title_end].strip()

    # Extract summary
    summary_start = xml.find('<summary>') + 9
    summary_end = xml.find('</summary>', summary_start)
    summary = xml[summary_start:summary_end].strip()

    # Extract publication date
    published_start = xml.find('<published>') + 11
    published_end = xml.find('</published>', published_start)
    published = xml[published_start:published_end].strip()

    # Extract authors
    authors = []
    pos = xml.find('<author>')
    while pos != -1:
        name_start = xml.find('<name>', pos) + 6
        name_end = xml.find('</name>', name_start)
        authors.append(xml[name_start:name_end].strip())
        pos = xml.find('<author>', name_end)
    
    # Format the extracted data as a string
    author_list = ', '.join(authors)
    data_string = f"Title: {title}\nSummary: {summary}\nPublished Date: {published}\nAuthors: {author_list}"
    
    return data_string


tools = [ bing_search,arxiv_search, get_time_and_date]

agent_executor = create_react_agent(model,tools=tools,state_modifier=template)


@csrf_exempt
def chat_with_openai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            user_input = data.get('message')
            if user_input:
               
                # system_template = "You are a helpful assistant. You are provided a set of tools to help you answer questions. You can search Arxiv or Bing for information. You can also chat with me."

                # prompt_template= ChatPromptTemplate.from_messages([
                #     ("system", system_template),
                #     ("user", "{text}"),
                # ])
                # chain = prompt_template | model | parser 
                # with_history = RunnableWithMessageHistory(chain,get_session_history,config=configuration)
                # response = with_history.invoke({"text": user_input})

                response=agent_executor.invoke({"messages": HumanMessage(content=user_input)})
                print(f"line 215:    {response}")
                # print("_____________________Break________________________\n")

                
                last_message = response['messages'][-1].content

                return JsonResponse({"response": last_message})
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
            model = data.get('model')
            client = ChatOpenAI(model=model, api_key=SECRET_KEY)
            if user_input:
                # Create message history for the model
                messages = [
                    SystemMessage(content="You are conversing about the topic: " + topic),
                ]
                if previous_message:
                    messages.append(HumanMessage(content=previous_message))
                messages.append(HumanMessage(content=user_input))
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