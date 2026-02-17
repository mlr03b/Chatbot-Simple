import os 
from dotenv import load_dotenv 
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

app = Flask(__name__)

# Tu configuración de Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY") 
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente servicial."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])
historial = ChatMessageHistory()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    usuario_input = request.json.get("message")
    prompt_completo = prompt.format_messages(history=historial.messages, input=usuario_input)
    respuesta = llm.invoke(prompt_completo)
    
    historial.add_user_message(usuario_input)
    historial.add_ai_message(respuesta.content)
    
    return jsonify({"response": respuesta.content})

if __name__ == "__main__":
    app.run(debug=True)