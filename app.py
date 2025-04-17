
# import os
# import time
# import csv
# from datetime import datetime
# import streamlit as st
# from dotenv import load_dotenv
# from pathlib import Path
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_models import ChatOllama
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load environment variables
# load_dotenv()
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

# # --- CSV Setup ---
# HISTORY_CSV = "history.csv"
# RESPONSES_CSV = "responses.csv"

# if not os.path.exists(HISTORY_CSV):
#     with open(HISTORY_CSV, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["user_id", "question_id", "question", "intent", "mode", "timestamp", "response_time", "similarity", "hallucination_score"])

# if not os.path.exists(RESPONSES_CSV):
#     with open(RESPONSES_CSV, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["question_id", "response", "similarity", "hallucination_score"])

# # --- Globals ---
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "mode" not in st.session_state:
#     st.session_state.mode = "freeform"
# if "qid" not in st.session_state:
#     st.session_state.qid = 1

# # --- Get retriever ---
# @st.cache_resource
# def get_retriever():
#     vector_db_path = Path(__file__).parent / 'data' / 'vector_db'
#     db = Chroma(
#         persist_directory=str(vector_db_path),
#         embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     )
#     retriever_instance = db.as_retriever()
#     retriever_instance.search_kwargs['k'] = 5
#     return retriever_instance

# retriever = get_retriever()

# # --- Intent classification ---
# def classify_intent(user_input: str) -> str:
#     input_lower = user_input.lower()
#     if any(word in input_lower for word in ["motivation", "character", "personality", "feelings"]):
#         return "character"
#     elif any(word in input_lower for word in ["plot", "story arc", "summarize", "summary", "main story"]):
#         return "plot"
#     elif any(word in input_lower for word in ["event", "incident", "happened", "occurred", "moment"]):
#         return "event"
#     else:
#         return "qa"

# # --- Prompt generator ---
# def create_prompt(intent: str, mode: str):
#     templates = {
#         "qa": {
#             "freeform": "Use the following context to answer the question.\n\n{context}\n\nQuestion: {input}",
#             "structured": "Using structured reasoning and the provided context, answer this query:\n\n{context}\n\nQuestion: {input}"
#         },
#         "plot": {
#             "freeform": "Here's some plot-related context:\n\n{context}\n\nNow answer this question: {input}",
#             "structured": "Given the plot context below, construct a structured response:\n\n{context}\n\nQuestion: {input}"
#         },
#         "character": {
#             "freeform": "Refer to the character info:\n\n{context}\n\nQ: {input}",
#             "structured": "Using the character traits and facts below, provide a structured answer:\n\n{context}\n\nQ: {input}"
#         },
#         "event": {
#             "freeform": "Given this event info:\n\n{context}\n\nAnswer: {input}",
#             "structured": "Based on the timeline and context of the event:\n\n{context}\n\nQuestion: {input}"
#         }
#     }
#     template = templates.get(intent, templates["qa"]).get(mode, templates["qa"]["freeform"])
#     return ChatPromptTemplate.from_template(template)

# # --- Similarity and Hallucination Scoring ---
# def compute_similarity(query: str, response: str) -> float:
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     query_embedding = embeddings.embed_query(query)
#     response_embedding = embeddings.embed_query(response)
    
#     similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
#     return similarity

# def compute_hallucination_score(response: str) -> float:
#     # Basic heuristic: longer responses with non-contextual content may have higher hallucination scores.
#     length_score = min(len(response) / 500, 1)  # Max length score of 1 for very long responses
#     return length_score  # Adjust this based on additional heuristics or model outputs.

# # --- Chat logic with logging ---
# def chat_with_bot(user_input, mode):
#     intent = classify_intent(user_input)
#     prompt = create_prompt(intent, mode)

#     # ✅ Use GPU by setting model_kwargs
#     llm = ChatOllama(model="mistral", model_kwargs={"num_gpu": 1})

#     doc_chain = create_stuff_documents_chain(llm, prompt)
#     chain = create_retrieval_chain(retriever, doc_chain)

#     start_time = time.time()
#     response = chain.invoke({"input": user_input})
#     end_time = time.time()

#     answer = response.get("answer", "Sorry, no answer was found.")
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     response_time = round(end_time - start_time, 3)
#     qid = st.session_state.qid

#     # Compute similarity and hallucination scores
#     similarity = compute_similarity(user_input, answer)
#     hallucination_score = compute_hallucination_score(answer)

#     st.session_state.history.append({
#         "user": user_input,
#         "bot": answer,
#         "intent": intent,
#         "mode": mode,
#         "similarity": similarity,
#         "hallucination_score": hallucination_score
#     })
#     st.session_state.qid += 1

#     # Write to history.csv
#     with open(HISTORY_CSV, mode='a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow([0, qid, user_input, intent, mode, timestamp, response_time, similarity, hallucination_score])

#     # Write to responses.csv
#     with open(RESPONSES_CSV, mode='a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow([qid, answer, similarity, hallucination_score])

#     return answer

# # --- Streamlit UI ---
# st.title("🧙 Harry Potter Chatbot")

# def mode_toggle_ui():
#     toggle = st.toggle("Structured Mode", value=(st.session_state.mode == "structured"))
#     st.session_state.mode = "structured" if toggle else "freeform"
#     mode_display = "Structured 🧠" if toggle else "Freeform ✨"
#     st.markdown(f"**Current Mode:** `{mode_display}`")

# st.markdown("### 📜 Conversation")
# for entry in st.session_state.history:
#     st.markdown(f"**You**: {entry['user']}")
#     st.markdown(f"**Bot** ({entry['mode'].capitalize()} - {entry['intent']}): {entry['bot']}")
#     st.write("---")

# mode_toggle_ui()

# with st.form("user_input_form", clear_on_submit=True):
#     user_input = st.text_input("Ask something about Harry Potter")
#     submitted = st.form_submit_button("Ask")
#     if submitted and user_input.strip():
#         chat_with_bot(user_input, st.session_state.mode)
#         st.rerun()
import os
import time
import csv
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# --- CSV Setup ---
HISTORY_CSV = "history.csv"
RESPONSES_CSV = "responses.csv"

if not os.path.exists(HISTORY_CSV):
    with open(HISTORY_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "question_id", "question", "intent", "mode", "timestamp", "response_time", "similarity", "hallucination_score", "irrelevancy_score"])

if not os.path.exists(RESPONSES_CSV):
    with open(RESPONSES_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "response", "similarity", "hallucination_score", "irrelevancy_score"])

# --- Globals ---
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "freeform"
if "qid" not in st.session_state:
    st.session_state.qid = 1

# --- Get retriever ---
@st.cache_resource
def get_retriever():
    vector_db_path = Path(__file__).parent / 'data' / 'vector_db'
    db = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    retriever_instance = db.as_retriever()
    retriever_instance.search_kwargs['k'] = 5
    return retriever_instance

retriever = get_retriever()

# --- Intent classification ---
def classify_intent(user_input: str) -> str:
    input_lower = user_input.lower()
    if any(word in input_lower for word in ["motivation", "character", "personality", "feelings"]):
        return "character"
    elif any(word in input_lower for word in ["plot", "story arc", "summarize", "summary", "main story"]):
        return "plot"
    elif any(word in input_lower for word in ["event", "incident", "happened", "occurred", "moment"]):
        return "event"
    else:
        return "qa"

# --- Prompt generator ---
def create_prompt(intent: str, mode: str):
    templates = {
        "qa": {
            "freeform": "Use the following context to answer the question.\n\n{context}\n\nQuestion: {input}",
            "structured": "Using structured reasoning and the provided context, answer this query:\n\n{context}\n\nQuestion: {input}"
        },
        "plot": {
            "freeform": "Here's some plot-related context:\n\n{context}\n\nNow answer this question: {input}",
            "structured": "Given the plot context below, construct a structured response:\n\n{context}\n\nQuestion: {input}"
        },
        "character": {
            "freeform": "Refer to the character info:\n\n{context}\n\nQ: {input}",
            "structured": "Using the character traits and facts below, provide a structured answer:\n\n{context}\n\nQ: {input}"
        },
        "event": {
            "freeform": "Given this event info:\n\n{context}\n\nAnswer: {input}",
            "structured": "Based on the timeline and context of the event:\n\n{context}\n\nQuestion: {input}"
        }
    }
    template = templates.get(intent, templates["qa"]).get(mode, templates["qa"]["freeform"])
    return ChatPromptTemplate.from_template(template)

# --- Scoring Functions ---
def compute_similarity(query: str, response: str) -> float:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_embedding = embeddings.embed_query(query)
    response_embedding = embeddings.embed_query(response)
    similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
    return similarity

def compute_hallucination_score(response: str) -> float:
    length_score = min(len(response) / 500, 1)
    return length_score

def compute_irrelevancy_score(user_input: str, response: str) -> float:
    similarity = compute_similarity(user_input, response)
    return 1 - similarity

# --- Chat logic with scoring and fallback ---
def chat_with_bot(user_input, mode):
    intent = classify_intent(user_input)
    prompt = create_prompt(intent, mode)

    llm = ChatOllama(model="mistral", model_kwargs={"num_gpu": 1})
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    start_time = time.time()
    response = chain.invoke({"input": user_input})
    end_time = time.time()

    answer = response.get("answer", "Sorry, no answer was found.")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response_time = round(end_time - start_time, 3)
    qid = st.session_state.qid

    similarity = compute_similarity(user_input, answer)
    hallucination_score = compute_hallucination_score(answer)
    irrelevancy_score = compute_irrelevancy_score(user_input, answer)

    # Fallback response if either score is below threshold
    # if hallucination_score < 0.75 or (irrelevancy_score) < 0.25:
    #     fallback_response = "I'm not confident in this answer. Let's try rephrasing or asking something else."
    #     answer = fallback_response

    st.session_state.history.append({
        "user": user_input,
        "bot": answer,
        "intent": intent,
        "mode": mode,
        "similarity": similarity,
        "hallucination_score": hallucination_score,
        "irrelevancy_score": irrelevancy_score
    })
    st.session_state.qid += 1

    with open(HISTORY_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([0, qid, user_input, intent, mode, timestamp, response_time, similarity, hallucination_score, irrelevancy_score])

    with open(RESPONSES_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([qid, answer, similarity, hallucination_score, irrelevancy_score])

    return answer

# --- Streamlit UI ---
st.title("🧙 Harry Potter Chatbot")

def mode_toggle_ui():
    toggle = st.toggle("Structured Mode", value=(st.session_state.mode == "structured"))
    st.session_state.mode = "structured" if toggle else "freeform"
    mode_display = "Structured 🧠" if toggle else "Freeform ✨"
    st.markdown(f"**Current Mode:** `{mode_display}`")

st.markdown("###  📜  Conversation")
for entry in st.session_state.history:
    st.markdown(f"**You**: {entry['user']}")
    st.markdown(f"**Bot** ({entry['mode'].capitalize()} - {entry['intent']}): {entry['bot']}")
    st.write("---")

mode_toggle_ui()

with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask something about Harry Potter")
    submitted = st.form_submit_button("Ask")
    if submitted and user_input.strip():
        chat_with_bot(user_input, st.session_state.mode)
        st.rerun()
