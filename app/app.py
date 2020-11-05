import os
from flask import Flask, request, jsonify, render_template
import pickle


import click
from flask import current_app, g
from flask.cli import with_appcontext
import os, uuid
#import azure.storage.blob as b
#from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)
try:
    print("Azure Blob storage v" + __version__ + " - Python quickstart sample")
    # Quick start code goes here
except Exception as ex:
    print('Exception:')
    print(ex)
app = Flask(__name__,static_url_path='/static')

#AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=gradadvqa;AccountKey=/+c+gEAhzS9ci5IbYowiwTxvi//2LZulTrkhWwYMB8OnMTEoiYWXhPwt4+LN+kY/HInI4KoQhICZC0G4nxCTlw==;EndpointSuffix=core.windows.net"
# Create the BlobServiceClient object which will be used to create a container client
#blob_service_client = b.BlobClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING,container_name='ml-model',blob_name='model1.pkl')
#downloader = blob_service_client.download_blob(0)

# Load to pickle
#b = downloader.readall()
#model = pickle.loads(b) 
    



@app.route("/") 
def home_view(): 
		return "Hello"

@app.route('/predict',methods=['POST','GET'])
def predict():
    to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What is this context?', 'id': '0'}]}]


    predictions, raw_outputs = model.predict(to_predict,3)
    return predictions[0]['answer'][0]

