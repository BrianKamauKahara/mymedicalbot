## STEP 01- Create a conda environment after opening the repository
python -m venv cuda
cuda/Scripts/activate

## STEP 02- install the requirements
Create a setup.py file in your master folder
pip install -r requirements.txt
Create a .env file in the root directory and add your Pinecone credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PINECONE_API_ENV = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
Download the quantize model from the link provided in model folder & keep the model in the model directory:
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
# run the following command
python store_index.py
# Finally run the following command
python app.py
Now,

open up localhost:
## Techstack Used:
Python
LangChain
Flask
Meta Llama2
Pinecone