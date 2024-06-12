from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
from pinecone import Pinecone

os.environ['PINECONE_API_KEY'] = '83655f48-c25b-4286-a2f1-146da0924de9'

pinecone_instance = Pinecone(api_key='83655f48-c25b-4286-a2f1-146da0924de9', environment='gcp-starter') #I dont think this line does anything but sijaitoa just incase

index_name = "mymedicalbot"


#STORING DATA INTO THE PINECONE DATABASE --ALREADY DONE
""" # Create a PineconeVectorStore instance with the Pinecone instance and index
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name) """

docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)

query = "What are Allergies"

docs=docsearch.similarity_search(query, k=3)
print(docs)








#OUTDATED CODE
""" pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)


index_name="medical-bot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name) """