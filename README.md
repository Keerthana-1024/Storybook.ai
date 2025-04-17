Here's a sample README for your project repo. I've based it on the code you provided:

---

# Harry Potter Chatbot

A conversational AI chatbot powered by **LangChain** and **ChatOllama** models to interact with users about the Harry Potter universe. It leverages **Chroma** for document retrieval and uses **HuggingFace embeddings** for semantic search to provide accurate, context-based answers.

---

## 🧙 Features

- **Intent Classification:** Classifies user queries into categories like `character`, `plot`, `event`, or `qa`.
- **Retrieval-Based Question Answering:** Uses document retrieval for precise answers.
- **Mode Toggle:** Switch between freeform or structured answer modes.
- **Similarity Scoring:** Computes response relevance using cosine similarity between user query and response.
- **Hallucination Scoring:** Scores the likelihood of hallucinated responses based on response length.
- **Irrelevancy Scoring:** Evaluates the degree of response irrelevance to the query.
- **History Logging:** Keeps track of all interactions in a CSV file for analytics.

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/harry-potter-chatbot.git
   cd harry-potter-chatbot
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file in the root directory and adding the following:

   ```
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_TRACING_V2=true
   ```

---

## 🛠️ Usage

1. Launch the app with **Streamlit**:

   ```bash
   streamlit run app.py
   ```

2. Interact with the chatbot through the Streamlit UI. You can ask questions about Harry Potter, and the bot will answer based on context retrieved from a database of documents.

3. You can toggle between **structured** and **freeform** modes to change how the bot answers.

---

## 💡 Intent Classification

The chatbot classifies user inputs into the following intents:

- `character`: Questions related to characters in the Harry Potter universe.
- `plot`: Questions related to the plot or story arc.
- `event`: Questions related to events or incidents.
- `qa`: General-purpose queries not fitting the above categories.

---

## ⚙️ How it Works

1. **Retrieval:** The bot retrieves contextually relevant information from the **Chroma vector database**.
2. **Intent Classification:** It classifies the user query into one of the predefined intents.
3. **Prompt Generation:** A dynamic prompt is created based on the intent and user mode (freeform or structured).
4. **LLM Interaction:** The chatbot uses the **ChatOllama model** to generate a response based on the prompt.
5. **Scoring:** Similarity, hallucination, and irrelevancy scores are calculated for each response.

---

## 🔧 Technologies Used

This project leverages the following technologies and libraries:

- **LangChain**: Utilized for chaining various components of the model.
- **ChatOllama**: A chatbot model optimized for seamless integration with LangChain.
- **Chroma**: A document retrieval system that enhances data access using embeddings.
- **HuggingFace Embeddings**: Used for transforming text into vector embeddings.
- **Streamlit**: Powers the user interface, enabling easy interaction with the model.

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---
