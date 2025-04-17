# import os
# import sys
# import pickle
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores.chroma import Chroma

# load_dotenv()
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"

# from langchain_community.vectorstores.chroma import Chroma

# def making_vector_db(documents: list, persist_directory: Path) -> Chroma:
#     db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=str(persist_directory))
#     return db

# def main():
#     current_path = Path(__file__)
#     root_path = current_path.parent.parent

#     processed_data_path = root_path / sys.argv[1]

#     vector_db_data_path = root_path / 'data' / 'vector_db'
#     vector_db_data_path.mkdir(exist_ok=True)

#     with open(processed_data_path, 'rb') as file:
#         documents = pickle.load(file)

#     db = making_vector_db(documents, vector_db_data_path)

#     print("Vector store has been successfully created and persisted at:", vector_db_data_path)

# if __name__ == "__main__":
#     main()
import os
import awsgi
import threading
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

retrieval_chain = None
retriever = None
lock = threading.Lock()

def load_vector_db(persist_directory: Path) -> Chroma:
    db = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return db

def create_prompt(mode: str):
    few_shots = {
        "plot": """
You are a story arc summarizer for the Harry Potter universe. Based only on the context, summarize the story progression relevant to the question.
Use structured HTML with <h3>Plot Summary</h3> and <p> tags.

Context:
<context>
{context}
</context>

Question:
<question>
{input}
</question>
""",
        "motivation": """
You are a character analyst. Based on the provided context, explain the character’s motivation in clear bullet points.
Use <h3>Character Motivation</h3> followed by <ul><li>...</li></ul>.

Context:
<context>
{context}
</context>

Question:
<question>
{input}
</question>
""",
        "event": """
You are an event explainer. Use the context to explain why the event happened, what it led to, and its consequences.
Format your answer in proper HTML using <h3>Event Explanation</h3> and <p>.

Context:
<context>
{context}
</context>

Question:
<question>
{input}
</question>
"""
    }
    return ChatPromptTemplate.from_template(few_shots.get(mode, few_shots["plot"]))

def initialize_retriever():
    global retrieval_chain, retriever
    if retriever is None:
        current_path = Path(__file__)
        root_path = current_path.parent
        vector_db_data_path = root_path / 'data' / 'vector_db'
        db = load_vector_db(persist_directory=vector_db_data_path)
        retriever = db.as_retriever()
        retriever.search_kwargs['k'] = 5

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ''
    if request.method == 'POST':
        initialize_retriever()
        user_input = request.form.get('input_text')
        mode = request.form.get('mode', 'plot')  # Default to 'plot' mode

        if user_input:
            with lock:
                llm = ChatOllama(model="mistral")  # Or llama2, gemma, etc.
                prompt = create_prompt(mode)
                document_chain = create_stuff_documents_chain(llm, prompt)
                chain = create_retrieval_chain(retriever, document_chain)

                response = chain.invoke({"input": user_input})
            answer = response.get('answer', 'Sorry, no answer was found.')
        else:
            answer = 'No input provided. Please enter a question.'
    return render_template('index.html', answer=answer)

def handler(event, context):
    return awsgi.response(app, event, context)

if __name__ == "__main__":
    initialize_retriever()
    app.run(debug=True, threaded=True)
