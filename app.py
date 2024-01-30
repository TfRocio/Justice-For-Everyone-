#Librerías
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Initialize your OpenAI chatbot
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Create a global variable to store the retrieval_chain
retrieval_chain = None

# Definimos un mensaje introductorio de nuestro Chatbot
system_intro = "Hola. Soy una asistente de AI con un extenso conocimiento sobre las regulaciones de tránsito de la CDMX."

# Definimos la historia del chat
chat_history = [
    HumanMessage(content="Puede un vehiculo circular con poliza de seguro vencida?"),
    AIMessage(content="No!")
]

# Input del usuario
user_input = "Dime por qué?"

# Función para procesar el input del usuario
def process_user_input(user_input):
# Remove leading and trailing whitespaces
    user_input = user_input.strip()

# Check if the user input is empty
    if not user_input:
        print("Error: Please provide a valid input.")
        return None

# Additional input processing logic if needed...

    return user_input

def invoke_chatbot(input_data, chat_history=None):
    try:
        # Get the system introduction
        system_intro = "You are an AI assistant with extensive knowledge of traffic regulations in Mexico City."

        # Generate the full input using the system introduction, chat history, and user input
        full_input = generate_full_input(input_data, system_intro, chat_history)
        # Prepare the input data with optional chat history
        input_payload = {"input": full_input}

        # Invoke the chatbot
        response = retrieval_chain.invoke(input_payload)

        # Extract and print the answer
        answer = response.get("answer", "Sorry, I couldn't generate a valid response.")
        print("Chatbot Response:", answer)

        return answer

    except Exception as e:
        print(f"An error occurred while invoking the chatbot: {str(e)}")
        # Handle the error gracefully, e.g., log the error, inform the user, or return an error message
        return "Sorry, an error occurred while processing your request."

def process_and_invoke(user_input, chat_history=None):
    processed_input = process_user_input(user_input)

    # Check if the input is valid before proceeding
    if processed_input is not None:
        # Generate the full input including system introduction and chat history
        full_input = generate_full_input(processed_input, system_intro, chat_history)
        invoke_chatbot(full_input, chat_history)

# Example usage
#process_and_invoke(user_input, chat_history)

# Template de la introducción y el input del usuario
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_intro),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

def create_retrieval_chain(retriever, document_chain):
    # ... (code for creating retrieval_chain)
    return retrieval_chain

# Carga del PDF
loader = PyPDFLoader("raw_data/Reglamento_CDMX.pdf")
pages = loader.load_and_split()

# Creación de vectores y embeddings
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(pages)
vector = FAISS.from_documents(documents, embeddings)

# Creación de la cadena de documentos
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# Creación del Retrieval Chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


def generate_full_input(user_input, system_intro, chat_history=None):
    # Combine the system introduction, chat history, and user input
    full_input = f"{system_intro}\n\n"
    if chat_history:
        for message in chat_history:
            full_input += f"{message.content}\n"
    full_input += f"User: {user_input}"
    return full_input

#app
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


import os
from langchain_openai import ChatOpenAI

# APP

import streamlit as st
import requests

# Título de la aplicación
st.title('Chatbot con Streamlit')

# Función para enviar y recibir mensajes del chatbot
def send_message(message):
    # URL de la API de tu chatbot
    url = 'URL_DE_TU_API'
    # Enviar la solicitud POST al chatbot con el mensaje del usuario
    response = requests.post(url, json={'message': message})
    # Devolver la respuesta del chatbot
    return response.json()['response']

# Crear una barra lateral para ingresar mensajes
user_input = st.text_input("Ingresa tu mensaje aquí:")

# Cuando se presiona Enter, enviar el mensaje al chatbot y mostrar la respuesta
if st.button("Enviar"):
    # Enviar el mensaje del usuario al chatbot y obtener la respuesta
    bot_response = send_message(user_input)
    # Mostrar la respuesta del chatbot
    st.text_area("Respuesta del Chatbot:", value=bot_response, height=200, max_chars=None, key=None)
