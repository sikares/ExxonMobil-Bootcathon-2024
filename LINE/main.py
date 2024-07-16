# -*- coding: utf-8 -*-

from flask import Flask,request
import os
import json
import requests
from dotenv import load_dotenv

#------------ end import zone -----------
load_dotenv()


token = os.getenv("LINE_TOKEN")
# print(token)

app = Flask(__name__)
# here
from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from cachetools import cached, LRUCache, TTLCache
# from langchain import PromptTemplate, GROQ_LLM, StrOutputParser, Memory, RunnablePassthrough


# CSV Import


from functools import lru_cache

docs_all = []

# @lru_cache(maxsize=None)
def load_documents():
    loader = DirectoryLoader("Data/", glob="**/*.txt")
    docs_all = loader.load()
    # print(type(docs_all))
    return docs_all

# Initialize the documents once, checking the WERKZEUG_RUN_MAIN environment variable
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    if len(docs_all) == 0:
        docs_all = load_documents()
        print("======= Loaded documents =======")
        print(len(docs_all))


    from langchain.text_splitter import RecursiveCharacterTextSplitter

    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(docs_all)

    # from langchain.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    from langchain_chroma import Chroma

    persist_directory = 'db'


    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=persist_directory)

from llm import invoke_model

chat_conver = {}




thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}



@app.route("/")
def hello():
    return "Hello World!"

#------------insert new code below --------
@app.route('/webhook' , methods = ['POST']) 
# for checking that the request 
def webhook():
    req = request.json
    print(req)
    if len(req["events"]) == 0:
        return '',200

    # replyToken = req['events'][0]['replyToken']

    handleRequest(req)
    return "", 200
    
# handle text incoming text
# the req of the events can be array but now we handle only 1 message at a time
def handleRequest(req):
    reply_url = 'https://api.line.me/v2/bot/message/reply'
    Authorization = 'Bearer {}'.format(token)

    headers = {'Content-Type':'application/json; charset=UTF-8','Authorization':Authorization}
    
    # now handle only 1 message that is sending from a user for now
    response = handleEvents(req["events"][0], req["destination"])
    replyToken = req['events'][0]['replyToken']
    
    data = json.dumps(
        {
            "replyToken":replyToken,
            "messages":[
                {
                    "type":"text",
                    "text":response,
                }
            ]
        }
    )
    r = requests.post(reply_url, headers=headers, data=data)
    print(r.text)

def handleEvents(event, destination):
    if destination not in chat_conver:
        chat_conver[destination] = []

    if event['message']['type'] == 'text':
        return handleMessage(event['message'], destination)
    elif event['message']['type'] == 'image':
        return handleImage(event['message'], destination)
    elif event['message']['type'] == 'audio':
        return handleAudio(event['message'], destination)
    else:
        print(f"Unknown event type: {event['type']}")
        return "sorry unknown type format we still can't handle this type of message"


# handle input type message
# return out the string that we want to send the user 
# out of the function to send a message

from langdetect import detect
from deep_translator import GoogleTranslator
# from googletrans import Translator
def handleMessage(event, destination):

    all_language = {
        'en': 'en',
        'th': 'th',
    }
    textFromUser = event['text']
    new_message = {
        "type": "msg",
        "msg": textFromUser
    }
    lang = detect(textFromUser)
    # output = Translator().translate(textFromUser, dest=f'{all_language[lang]}').text
    
    print("chat convert destination")
    print(chat_conver[destination])
    # translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")
    if len(chat_conver[destination]) == 0 :
        # Initialize chat_conver[destination] if it doesn't exist
        chat_conver[destination] = []
        chat_conver[destination].append(new_message)
        inputs = {"initial_question": textFromUser, "num_steps": 0}
        result = invoke_model(inputs)
        print("result returning from latest message text", result)
        if lang == 'th':
            return GoogleTranslator(source='auto', target='th').translate(result['final_answer'])
        else:
            return result['final_answer']
    else:
        latest_message_type = chat_conver[destination][-1]["type"]
        
        if latest_message_type == "car":
            car_info = chat_conver[destination][-1]["msg"]
            chat_conver[destination].append(new_message)
            print("======== CAR INFO =========")
            print(car_info)
            initialPrompt = textFromUser + car_info
            inputs = {"initial_question": initialPrompt, "num_steps": 0}
            result = invoke_model(inputs)
            print("result returning from latest message car", result)
            if lang == 'th':
                return GoogleTranslator(source='auto', target='th').translate(result['final_answer'])
            else:
                return result['final_answer']
        else:
            chat_conver[destination].append(new_message)
            inputs = {"initial_question": textFromUser, "num_steps": 0}
            result = invoke_model(inputs)
            print("result returning from latest message text", result)
            if lang == 'th':
                return GoogleTranslator(source='auto', target='th').translate(result['final_answer'])
            else:
                return result['final_answer']

    
    # return "receiving this from handle message" + textFromUser
    # print(textFromUser)
    # return invoke_model(textFromUser)



# handle input type image
# return out the string that we want to send the user 
# out of the function to send a message
def handleImage(event, destination):
    
    # line doesn't allow to get the image data directly but need to call an api instead
    messageId = event["id"]
    getMsgContentUrl = f"https://api-data.line.me/v2/bot/message/{messageId}/content"
    
    Authorization = 'Bearer {}'.format(token)

    headers = {'Content-Type':'application/json; charset=UTF-8','Authorization':Authorization}
    # Returns status code 200 and the content in binary
    response = requests.request("GET", getMsgContentUrl, headers=headers)
    content_type = response.headers['Content-Type']

    if response.status_code == 200:
        image_data = response.content
        responseFromCarnet = recognize_file(messageId, image_data, content_type)
        if 'error' in responseFromCarnet:
            return "can't process your image cause of " + responseFromCarnet['error']
        
        car_info = responseFromCarnet["car"]
        if destination not in chat_conver:
            chat_conver[destination] = []
            
        string_result = f"""Car Information:\nMake: {car_info["make"]}\nModel: {car_info["model"]}\nGeneration: {car_info["generation"]}\nYears: {car_info["years"]}\nProbability: {car_info["prob"]}%"""
        string_result2 = f"""Make: {car_info["make"]} Model: {car_info["model"]} Generation: {car_info["generation"]} Years: {car_info["years"]}"""

        new_message = {
            "type": "car",
            "msg": string_result2.strip()
        }

        chat_conver[destination].append(new_message)
        string_result2 = f"""I have recognized the car image for you
Car Information:
Make: {car_info["make"]}
Model: {car_info["model"]}
Generation: {car_info["generation"]}
Years: {car_info["years"]}
Probability: {car_info["prob"]}%"""
        # return "I have recognized the car image for you\n" + string_result
        return string_result2

    
    else:
        return "Get image error: " + response.text

import os
import tempfile
import requests
from groq import Groq

def transcribe_audio_groq(audio_file_path):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with open(audio_file_path, "rb") as file:
        translation = client.audio.translations.create(
            file=(audio_file_path, file.read()),
            model="whisper-large-v3",
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            temperature=0.0  # Optional
        )
    return translation.text

# handle input audio
def handleAudio(event, destination):
    messageId = event["id"]
    getMsgContentUrl = f"https://api-data.line.me/v2/bot/message/{messageId}/content"
    
    Authorization = 'Bearer {}'.format(token)

    headers = {'Content-Type':'application/json; charset=UTF-8','Authorization':Authorization}
    response = requests.request("GET", getMsgContentUrl, headers=headers)
    if response.status_code == 200:
        binaryAudio = response.content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(binaryAudio)
            temp_audio_file_path = temp_audio_file.name

        audioText = transcribe_audio_groq(temp_audio_file_path)
        new_message = {
            "type": "audio",
            "msg": audioText
        }
        chat_conver[destination].append(new_message)
        inputs = {"initial_question": audioText, "num_steps": 0}
        result = invoke_model(inputs)
        print("result returning from latest message text", result)
        return f"Transcription: {audioText} \n" + result['final_answer']
    else:
        return "Can't process your audio"




# version to use with a lineOa
# change at imageFile directly send the file_type to the files
import requests

def recognize_file(file_name, file_binary, file_type):
    url = "https://carnet.ai/recognize-file"
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,th;q=0.8",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "Referer": "https://carnet.ai/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

    files = {
        "imageFile": (file_name, file_binary, file_type)
    }
    
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error": "Invalid JSON response"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "Internal exception occurred"}



#------------ end edit zone  --------
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT','5000')), use_reloader=False, threaded=False)