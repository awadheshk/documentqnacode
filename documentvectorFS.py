!pip install --quiet --upgrade google_cloud_firestore google_cloud_aiplatform langchain langchain-google-vertexai langchain_community langchain_experimental pymupdf google-cloud-logging

## Library loading
# GCP Vertex AI functionalities
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting,  HarmBlockThreshold, HarmCategory

# Loading and saving of serialized data
import pickle

#Dispalaying Markdown or formatted content
from IPython.display import display, Markdown

from langchain_google_vertexai import VertexAIEmbeddings  # Embedding
from langchain_community.document_loaders import PyMuPDFLoader  # pdf  loader
from langchain_experimental.text_splitter import SemanticChunker # pdf document processer or splitter

# Firestore database (Vector database)
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure  # Ecludian distance

# Flask UI related library
import os
import json
import logging
import google.cloud.logging
from flask import Flask, render_template, request

## Project and environment setting
# PROJECT_ID=(!gcloud info --format='value(config.project)')
PROJECT_ID = "qwiklabs-gcp-02-7ce16daaee58"

#ZONE=$(!gcloud compute project-info describe \
#--format="value(commonInstanceMetadata.items[google-compute-default-zone])")
# LOCATION="${ZONE%-*}"
LOCATION = "us-central1"

#print("PROJECT ID:", PROJECT_ID)
#print("Location:",LOCATION )

# Model initiaization

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

## Loading and cleaning pdf document
#!gsutil mb gs://mydocument1-bucket1

# !gsutil cp <local_file_path> gs://<bucket_name>/<object_name>
#!gsutil cp "C:\\Users\\A585918\\Downloads\\google-2023-carbon-removal-research-award.pdf"  . # gs://mydocument1-bucket1/

#!gcloud storage cp gs://mydocument1-bucket1/google-2023-carbon-removal-research-award.pdf .

loader = PyMuPDFLoader("./google-2023-carbon-removal-research-award.pdf")
data = loader.load()

def clean_page(page):
  return page.page_content.replace("-\n","")\
                          .replace("\n"," ")\
                          .replace("\x02","")\
                          .replace("\x03","")\
                          .replace("fo d P R O T E C T I O N  T R A I N I N G  M A N U A L","")\
                          .replace("N E W  Y O R K  C I T Y  D E P A R T M E N T  O F  H E A L T H  &  M E N T A L  H Y G I E N E","")

## Chunking of pdf and creating embedding
cleaned_pages = []
for pages in data:
  cleaned_pages.append(clean_page(pages))

text_splitter = SemanticChunker(embedding_model)
docs = text_splitter.create_documents(cleaned_pages[0:4])     # First 5 pages of the pdf
chunked_content = [doc.page_content for doc in docs]
chunked_embeddings = embedding_model.embed_documents(chunked_content)

# Flask init
# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Application Variables
BOTNAME = "Custom document QnA Bot"
SUBTITLE = "Custom document questionnaire or searcher"

app = Flask(__name__)

## Storing the embedding in Vector database (Firestore database)
db = firestore.Client(project=PROJECT_ID)
collection = db.collection('mydatacollection')

for i, (content, embedding) in enumerate(zip(chunked_content, chunked_embeddings)):
    doc_ref = collection.document(f"doc_{i}")
    doc_ref.set({
        "content": content,
        "embedding": Vector(embedding)
    })

# indexing document in DB
!gcloud firestore indexes composite create --project=qwiklabs-gcp-02-7ce16daaee58 --collection-group=mydatacollection --query-scope=COLLECTION --field-config=vector-config='{"dimension":"768","flat": "{}"}',field-path=embedding

## Vector search method using Eucleadian distance measure method

def search_vector_database(query: str):
  query_embedding = embedding_model.embed_query(query)
  vector_query = collection.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_embedding),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=5,
  )
  docs = vector_query.stream()
  # context = [result.to_dict()['content'] for result in docs]
  context = " ".join([result.to_dict()['content'] for result in docs])
  return context

# Vertex AI Generative Model Initialization

gen_model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config={"temperature": 0},
)

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        # method=gen_model.HarmBlockMethod.PROBABILITY, #PROBABLITY
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
  ),
]

## main

# Function to Send Query and Context to Gemini and Get the Response
def ask_gemini(question):
    # Create a prompt template with context instructions
    prompt_template = "Using the context provided below, answer the following question:\nContext: {context}\nQuestion: {question}\nAnswer:"

    # Retrieve context for the question using the search_vector_database function
    context = search_vector_database(question)

    # Format the prompt with the question and retrieved context
    formatted_prompt = prompt_template.format(context=context, question=question)

    # Define the generation configuration for the Gemini model
    generation_config = GenerationConfig(
        temperature=0.7,  # Adjust temperature as needed
        max_output_tokens=256,  # Maximum tokens in the response
        response_mime_type="application/json",  # MIME type of response
    )

    # Define the contents parameter with the prompt text
    contents = [
        {
            "role": "user",
            "parts": [{"text": formatted_prompt}]
        }
    ]

    # Call the generate_content function with the defined parameters
    response = gen_model.generate_content(
        contents=contents,
        generation_config=generation_config
    )

    # Parse the JSON response to extract the answer field
    response_text = response.text if response else "{}"  # Default to empty JSON if no response
    try:
        response_json = json.loads(response_text)  # Parse the JSON string into a dictionary
        answer = response_json.get("answer", "No answer found.")  # Get the "answer" field
    except json.JSONDecodeError:
        answer = "Error: Unable to parse response."

    return answer

search_vector_database("What are areas of investigation would benefit from additional scientific support?")

# Home page route
@app.route("/", methods=["POST", "GET"])
def main():
    # Initial message for GET request
    if request.method == "GET":
        question = ""
        answer = "Hello, I'm document questionnaire bot, what can I do for you?"

    # Handle POST request when the user submits a question
    else:
        question = request.form["input"]

        # Log the user's question
        logging.info(question, extra={"labels": {"service": "documentqna-service", "component": "question"}})

        # Get the answer from Gemini based on the vector database search
        answer = ask_gemini(question)

    # Log the generated answer
    logging.info(answer, extra={"labels": {"service": "documentqna-service", "component": "answer"}})
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": "Document QnA Bot",
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
  
