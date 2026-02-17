import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq # Proveedor que se vaya a usar
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

# 1. Inicialización del Modelo LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 2. Definición de la estructura que une Prompt con el historial (Une el pasado y el presente)
     #(PREPARACIÓN DE LA PLANTILLA/MOLDE)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente servicial."), # Damos la personalidad al sistema (Ej.Si pongo "Respondeme en rimas", me responderá en rimas)
    MessagesPlaceholder(variable_name="history"), #Creamos un espacio para el historial
    ("user", "{input}") # Creamos un espacio reservado para la pregunta del usuario
])

# 3. Se crea una lista vacía donde se irán almacenando las diferentes preguntas hechas por el usuario y respuestas dadas por la IA
historial = ChatMessageHistory()

# 4. BUCLE DEL CHATBOT 
print("Chatbot iniciado. Escribe 'salir' para terminar.")

while True:
    #4.1. Se pide al usuario que ingrese por teclado 
    usuario = input("Tu: ")

    #4.1.1. Si el usuario pide "salir" se sale del bucle
    if usuario.lower() == "salir":
        break

    # 4.2. Preparamos el mensaje que se envía a la IA con el input (pregunta que queremos que responda) e historial (Pegando los mensajes anteriores para que la IA tenga memoria)
           #(LLENADO DE LA PLANTILLA/MOLDE)
    prompt_completo = prompt.format_messages(
        history=historial.messages, 
        input=usuario
    )

    # 4.3. Se envía el mensaje al modelo para recibir una respuesta (llm.invoke(pregunta) devuelve la respuesta de la pregunta hecha al modelo)
    respuesta = llm.invoke(prompt_completo)

    # 4.4. Se guarda la pregunta y respuesta en el historial
    historial.add_user_message(usuario) # Guarda la pregunta
    historial.add_ai_message(respuesta.content) # Guarda la respuesta

    # 4.5. Se imprime la respuesta dada por la IA
    print(f"IA: {respuesta.content}")