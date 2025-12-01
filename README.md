# **mAI**
Grounded **RAG GenAI pipeline** using **LangChain**, **LangGraph**, **Qdrant**, **Hugging Face**, and **Gemini via Portkey**.
Includes **semantic + keyword fallback retrieval**, HuggingFace embeddings, and **Gradio UI**.
Orchestrates RAG flow with **confidence threshold** and **context-aware answers**.

---

## ✅ **Steps to Run**

### 1. **Install dependencies**
```bash
%pip install -r requirements.txt
```

---

### 2. **Set up API keys**
Make sure you have:

- **Portkey API Key**
- **Google Gemini API Key**

Run:
```bash
python setup_keys.py
```
This script will prompt you to upload or set your keys.

---

### 3. **Start the app**
```bash
python agent.py
```
This will launch the **Gradio UI**. Click the link to interact with the app.

---

## 📂 **Repo Structure**
```
setup_keys.py      # Handles environment setup
agent.py           # Starts the app and UI
requirements.txt   # Dependencies
```

---

## ✅ **Features**
- Semantic + keyword fallback retrieval
- Confidence threshold (default: `0.3`)
- Context-aware answers with sources
- Fallback to **“I don’t know”** when confidence is low

---

## 📖 **Blog**
Read the detailed explanation here: *[Add your blog link]*
