import re
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from markupsafe import Markup
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
import numpy as np
from numpy.linalg import norm
import pandas as pd
import faiss
import os
from dotenv import load_dotenv
import markdown

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Add nl2br filter to Jinja2 environment using flask.Markup
app.jinja_env.filters['nl2br'] = lambda value: Markup(value.replace('\n', '<br>'))

# ---------------------- Load Models and Data ----------------------
df = pd.read_excel("department_marketing_mix_data.xlsx")
df["text"] = df.apply(lambda row: f"{row['Department']} department ran {row['Channel']} campaign in {row['Year']}. Spend: {row['Spend']}, ROI: {row['ROI']}, Incremental ROI: {row['Incremental ROI']}", axis=1)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment. Please set it in your .env file.")

EMBEDDING_MODEL_NAME = "mistral-embed"
EMBEDDING_DIM = 1024  # Mistral embed model default dimension

embed_model = MistralAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# Precompute and normalize document embeddings for FAISS
document_texts = list(df["text"])
raw_doc_embeddings = embed_model.embed_documents(document_texts)
doc_embeddings = [np.array(e) / norm(e) for e in raw_doc_embeddings]
doc_embeddings_np = np.stack(doc_embeddings)
if doc_embeddings_np.dtype != np.float32:
    doc_embeddings_np = doc_embeddings_np.astype(np.float32)
index = faiss.IndexFlatL2(doc_embeddings_np.shape[1])
index.add(doc_embeddings_np)

LANGUAGE_MODEL = ChatMistralAI(model="mistral-large-latest")

PROMPT_TEMPLATE = """
You are an agentic AI assistant specialized in marketing mix optimization for a pharmaceutical company.
You have access to past marketing mix data as context.
When given a user query containing constraints, goals, budget, channel, or other inputs, analyze these carefully and provide optimized marketing tactic recommendations.
If the query is a greeting, casual conversation, or unrelated to the context, respond appropriately, naturally and within 2-3 lines only.
For example, if user says hello, you can say "Hello <user name> from <role>! How can I assist you with your marketing mix optimization today?"
Use the provided context only when it is relevant to the query.
If the context is insufficient to answer the query confidently, say so and ask the user a clarification question.
If historical data for a required channel is missing, extrapolate or reason based on available data and clearly mention this in your answer.
Keep your response concise and clear. For questions related to planning, explain your reasoning in your response in a clear, concise and to the point manner.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def format_answer_for_display(answer):
    try:
        # Convert Markdown to HTML
        html_output = markdown.markdown(answer, extensions=['extra', 'smarty'])
        return html_output
    except Exception as e:
        print(f"Markdown parsing error: {e}")
        return answer

def retrieve_similar_documents(query, top_k=5):
    departments = df['Department'].str.lower().unique()
    channels = df['Channel'].str.lower().unique()
    year_match = re.search(r'(20\d{2})', query)
    year = int(year_match.group(1)) if year_match else None

    # Use session['role'] as department if not mentioned in query
    department = None
    for dep in departments:
        if dep in query.lower():
            department = dep
            break
    if department is None and 'role' in session:
        department = session['role'].lower()

    mentioned_channels = [ch for ch in channels if ch in query.lower()]
    wants_all_channels = bool(re.search(r'all channels', query, re.IGNORECASE))

    filtered = df.copy()
    if department:
        filtered = filtered[filtered['Department'].str.lower() == department]
    if year:
        filtered = filtered[filtered['Year'] == year]

    # If user wants all channels, ignore channel filtering entirely
    if wants_all_channels:
        pass  # do not filter by channel
    elif mentioned_channels:
        filtered = filtered[filtered['Channel'].str.lower().isin(mentioned_channels)]

    # If budget/objective is mentioned, prioritize rows with spend/ROI
    if re.search(r'budget|roi|spend|invest|objective', query, re.IGNORECASE):
        filtered = filtered.sort_values(by=['ROI', 'Spend'], ascending=False)

    docs = [Document(page_content=row['text']) for _, row in filtered.iterrows()]
    # If no docs found, fallback to semantic search using normalized embeddings
    if not docs:
        query_embedding_raw = embed_model.embed_query(query)
        query_embedding = np.array(query_embedding_raw) / norm(query_embedding_raw)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
        docs = [Document(page_content=df.iloc[idx]["text"]) for idx in I[0]]
    # Add a flag if a required channel is missing
    required_channels = []
    if 'email marketing' in query.lower():
        required_channels.append('email marketing')
    missing_channels = [ch for ch in required_channels if ch not in filtered['Channel'].str.lower().unique()]
    if missing_channels:
        docs.append(Document(page_content=f"WARNING: Historical data for channels {', '.join(missing_channels)} is missing. Please extrapolate or reason based on available data."))
    return docs

def generate_answer(user_query, context_documents, user_role, history=None):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    history_text = "\n".join(history) if history else ""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format_prompt(
        user_query=user_query,
        document_context=context_text,
        user_role=user_role,
        history=history_text
    )
    response = LANGUAGE_MODEL.invoke(formatted_prompt.to_string())
    response_content = response.content
    print(response_content)
    if "</think>" in response_content:
        ans_split = response_content.split("</think>")
        print(ans_split)
        return ans_split[1].strip()
    return response_content.strip()

@app.route("/", methods=["GET"])
def signin_get():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signin_post():
    username = request.form.get("username")
    role = request.form.get("role")
    session["username"] = username
    session["role"] = role
    return redirect(url_for("chat_get"))

@app.route("/chat", methods=["GET"])
def chat_get():
    if "username" not in session or "role" not in session:
        return redirect(url_for("signin_get"))
    return render_template("chat.html", username=session["username"], role=session["role"])

@app.route("/chat", methods=["POST"])
def chat_post():
    if "username" not in session or "role" not in session:
        return redirect(url_for("signin_get"))
    query = request.form.get("query")
    # Store conversation history in session
    if "history" not in session:
        session["history"] = []
    session["history"].append(f"User: {query}")
    docs = retrieve_similar_documents(query)
    if not docs:
        answer = "Not enough relevant data found. Please rephrase or provide more context."
    else:
        print("generating answer")
        answer = generate_answer(query, docs, session["role"], history=session["history"])
        print("answer generated")
        if not isinstance(answer, str):
            answer = str(answer)
        # Format answer for display
        try:
            answer = format_answer_for_display(answer)
        except Exception as e:
            print(f"Formatting error: {e}")
            # Fallback to plain answer
    session["history"].append(f"Agent: {answer}")
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"answer": answer})
    else:
        return render_template("chat.html", username=session["username"], role=session["role"], query=query, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)