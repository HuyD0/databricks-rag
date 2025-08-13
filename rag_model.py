# rag_model.py

import mlflow
import os
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


load_dotenv()

class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config):
        """
        Stores the configuration for the model.
        """
        self.config = config

    def load_context(self, context):
        """
        This method is called once when the model is loaded for serving.
        It initializes all necessary clients and objects using the config.
        """
        # DefaultAzureCredential will automatically use the environment variables
        # set for the service principal (AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET)
        credential = DefaultAzureCredential()
        
        token_provider = get_bearer_token_provider(
            credential, 
            os.getenv("COGNITIVE_SERVICES_SCOPE", "https://cognitiveservices.azure.com/.default")
        )

        # Initialize clients using the configuration passed during instantiation
        self.embedding_client = AzureOpenAI(
            api_version=self.config["OPENAI_API_VERSION"],
            azure_endpoint=self.config["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider,
        )

        self.chat_client = AzureOpenAI(
            api_version=self.config["OPENAI_API_VERSION"],
            azure_endpoint=self.config["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider,
        )

        self.search_client = SearchClient(
            endpoint=self.config["AZURE_SEARCH_ENDPOINT"],
            index_name=self.config["AZURE_SEARCH_INDEX_NAME"],
            credential=credential
        )
        
        self.system_message = self.config.get("SYSTEM_MESSAGE_PROMPT", "You are a helpful assistant.")
        self.chat_deployment = self.config["CHAT_MODEL_DEPLOYMENT"]
        self.embedding_deployment = self.config["EMBEDDING_MODEL_NAME"]
        self.temperature = float(self.config.get("GENERATION_TEMPERATURE", 0.1))


    def _retrieve_documents(self, query, top_k):
        """
        Internal method to perform the retrieval step.
        """
        embedding_response = self.embedding_client.embeddings.create(
            model=self.embedding_deployment, 
            input=query
        )
        query_vector = embedding_response.data[0].embedding
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        results = self.search_client.search(
            select=["source", "page_title", "content"],
            vector_queries=[vector_query],
            top=top_k
        )
        return [{"source": res["source"], "page_title": res["page_title"], "content": res["content"]} for res in results]

    def _generate_answer(self, query, retrieved_docs):
        """
        Internal method to perform the generation step.
        """
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        user_message = f"CONTEXT:\n---\n{context}\n---\nQUESTION: {query}"
        
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": user_message}]
        
        chat_response = self.chat_client.chat.completions.create(
            model=self.chat_deployment, 
            messages=messages, 
            temperature=self.temperature
        )
        return chat_response.choices[0].message.content

    def predict(self, context, model_input):
        """
        This is the main prediction method called by MLflow.
        """
        questions = model_input["question"]
        # Use a default top_k from config if not provided in the input DataFrame
        default_top_k = int(self.config.get("TOP_K", 5))
        top_k_values = model_input.get("top_k", [default_top_k] * len(questions))

        answers = []
        for i, question in enumerate(questions):
            documents = self._retrieve_documents(question, top_k_values[i])
            final_answer = self._generate_answer(question, documents)
            answers.append(final_answer)

        return pd.DataFrame({"answer": answers})
    
# --- Model Instantiation for MLflow ---
# This block runs when MLflow loads the model from the code_path.
# It reads all necessary configurations from environment variables.

print("Initializing model configuration from environment variables...")
model_config = {
    # Read all required configs from the environment
    "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "AZURE_SEARCH_INDEX_NAME": os.getenv("AZURE_SEARCH_INDEX_NAME"),
    "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME"),
    "EMBEDDING_ENDPOINT": os.getenv("EMBEDDING_ENDPOINT"),
    "CHAT_MODEL_DEPLOYMENT": os.getenv("CHAT_MODEL_DEPLOYMENT"),
    "SYSTEM_MESSAGE_PROMPT": os.getenv("SYSTEM_MESSAGE_PROMPT"),
    "TOP_K": os.getenv("TOP_K"),
    "GENERATION_TEMPERATURE": os.getenv("GENERATION_TEMPERATURE")
}

# Instantiate the model with the loaded configuration
rag_model_instance = RAGModel(config=model_config)

# Set the model instance for MLflow to discover
mlflow.models.set_model(rag_model_instance)
