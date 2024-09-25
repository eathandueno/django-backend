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
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
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
from bs4 import BeautifulSoup
from scholarly import scholarly
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from transformers import pipeline
search = DuckDuckGoSearchRun()
hugging_face = config('HUGGINGFACEHUB_API_TOKEN')
store = {}
configuration = {"configurable":{"session_id":"testing_session"}}
SECRET_KEY = config('OPENAI_API_KEY')
bing_key = config('BING_API_KEY')

model = ChatOpenAI(model="gpt-4o", api_key=SECRET_KEY)
parser = StrOutputParser()

controller_template = """Verify if the task was completed: {messages} """

questioner_template = """
"""

template = """Answer the following questions as best you can. You have access to the following tools:

- Bing Search: Search Bing for information.
- Arxiv Search: Search Arxiv for papers.
- Get Time and Date: Retrieve the current date and time.
- Scrape Website: Scrape a website for information.
- Search Google Scholar: Search Google Scholar for academic papers.
- DuckDuckGo Search: Search DuckDuckGo for information.

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of bing_search, search_google_scholar, arxiv_search, get_time_and_date, scrape_website, duckduckgo_search

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
def bing_search(query="", searchType="WebPages", count=1):
    """
    Search Bing and retrieve the top `count` results, presenting relevant snippets with citations.

    Args:
        query (str): The text query for which search results are desired.
        searchType (str): The category of search results to be retrieved.
        count (int): Number of search results to retrieve.
    
    Returns:
        str: A string of the cleaned snippets of the defined number of search results with citations.
    """
    search_results = make_bing_request(query, searchType, count)
    snippets_with_citations = extract_snippets_with_citations(search_results, searchType)
    
    return format_snippets(snippets_with_citations)

def make_bing_request(query, searchType, count):
    
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
    return response.json()

def extract_snippets_with_citations(search_results, searchType):
    snippets_with_citations = []
    if searchType.lower() == "webpages" and "webPages" in search_results:
        for result in search_results["webPages"]['value']:
            snippet = remove_html_tags(result['snippet'])
            citation = result['url']
            snippets_with_citations.append(f"{snippet} (Source: {citation})")
    elif searchType.lower() == "news" and "news" in search_results:
        for tempArticle in search_results["news"]['value']:
            article = extract_news_details(tempArticle)
            snippet = remove_html_tags(article['title'])
            citation = article['url']
            snippets_with_citations.append(f"{snippet} (Source: {citation})")
    # Additional search types can be handled similarly
    else:
        snippets_with_citations.append("No results found")
    
    return snippets_with_citations

def format_snippets(snippets_with_citations):
    return " \n ".join(snippets_with_citations)

def extract_news_details(json_data):
    article_details = {
        'title': json_data.get('name', '').replace('<b>', '').replace('</b>', ''),
        'url': json_data.get('url', ''),
        'description': json_data.get('description', ''),
        'image_url': json_data.get('image', {}).get('contentUrl', ''),
        'thumbnail_url': json_data.get('image', {}).get('thumbnail', {}).get('contentUrl', ''),
        'date_published': json_data.get('datePublished', ''),
        'provider': json_data.get('provider', [{}])[0].get('name', ''),
    }
    return article_details

def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)  # This regex matches any text within <>, including the brackets
    return clean_text

@tool
def duckduckgo_search(query: str) -> str:
    """
    Searches DuckDuckGo with the given query string.

    Args:
        query (str): The search query string.

    Returns:
        str: The result of the search.
    """
    result = search.invoke(query)
    return result

@tool
def search_google_scholar(query):
    """
    Search Google Scholar for the specified query and retrieve the first result.

    Args:
        query (str): The query to search for on Google Scholar.

    Returns:
        str: A string containing the title, authors, year, abstract, and URL of the first result.
    """
    # Search for the query on Google Scholar
    search_query = scholarly.search_pubs(query)
    
    # Retrieve the first result
    result = next(search_query)
    
    # Prepare the output string
    result_string = f"Title: {result['bib']['title']}\n" \
                    f"Authors: {', '.join(result['bib']['author'])}\n" \
                    f"Year: {result['bib']['pub_year']}\n" \
                    f"Abstract: {result.get('abstract', 'No abstract available')}\n" \
                    f"URL: {result.get('pub_url', 'No URL available')}"
    
    return result_string

@tool
def scrape_website(url: str) -> str:
    """
    Scrapes the specified URL and extracts the main details such as title, 
    meta description, and the main content.

    Parameters:
        url (str): The URL of the website to scrape.

    Returns:
        str: A string formatted as JSON containing the scraped data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the title
        title = soup.title.string if soup.title else None
        
        # Extract the meta description
        meta_description = None
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_description = meta_tag.get('content', None)
        
        # Extract the main content (this is a generic approach, and may vary depending on the site)
        main_content = None
        if soup.find('article'):
            main_content = soup.find('article').get_text(strip=True)
        elif soup.find('div', attrs={'id': 'main-content'}):
            main_content = soup.find('div', attrs={'id': 'main-content'}).get_text(strip=True)
        elif soup.body:
            main_content = soup.body.get_text(strip=True)
        
        # Compile the scraped data
        scraped_data = {
            "title": title,
            "meta_description": meta_description,
            "main_content": main_content
        }
        
        return json.dumps(scraped_data, indent=4)
    except requests.exceptions.RequestException as e:
        return f"Error occurred while scraping the website: {e}"

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


tools = [get_time_and_date, bing_search, search_google_scholar, arxiv_search, get_time_and_date, scrape_website, duckduckgo_search]
question_tools = [get_time_and_date, bing_search, search_google_scholar, arxiv_search, get_time_and_date, scrape_website, duckduckgo_search]


config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()

agent_questioner = create_react_agent(model,tools=question_tools,state_modifier=questioner_template,checkpointer=memory)
task=pipeline(model="google/flan-t5-xl")
@csrf_exempt
def chat_with_openai(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON from the request body
            user_input = data.get('message')
            if user_input:           
                steps = []
                chain =   model | parser 
               
                # response=task(f"Answer the following user query: {user_input}")
                # print(response[0]['generated_text'])
                task_provider = agent_questioner.invoke({"messages": """Answer the following questions as best you can. You have access to the following tools:
                Bing Search, Arxiv Search, Scrape Website and Get Current date and time, Google Scholar Search, DuckDuckGo Search
                Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of Bing Search, Arxiv Search, Scrape Website and Get Current date and time, Google Scholar Search, DuckDuckGo Search
                Action Input: the input to the action
                Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question in JSON format
Begin!
                QUESTION: {user_input}  .\n
                SYSTEM:
                """.format(user_input=user_input)},config=config)
                task_response = task_provider
                for message in task_provider['messages']:
                    steps.append(f"{getattr(message, 'tool_calls', f'{type(message)}')}   \n  -     {message.content}")
                last_message = task_response['messages'][-1].content
                # print(last_message)
                # summary = chain.invoke(f"Please format the following into structure output: {last_message}")
                return JsonResponse({"response": last_message, "steps": steps})
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