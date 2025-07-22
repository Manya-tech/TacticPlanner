from flask import Flask, render_template, request, redirect, url_for, session
from flask import jsonify
from markupsafe import Markup
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import faiss
import jinja2
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Add nl2br filter to Jinja2 environment using flask.Markup
app.jinja_env.filters['nl2br'] = lambda value: Markup(value.replace('\n', '<br>'))

# ---------------------- Load Models and Data ----------------------
df = pd.read_excel("department_marketing_mix_data.xlsx")
df["text"] = df.apply(lambda row: f"{row['Department']} department ran {row['Channel']} campaign in {row['Year']}. Spend: {row['Spend']}, ROI: {row['ROI']}, Incremental ROI: {row['Incremental ROI']}", axis=1)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_mmm_index.index")

LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are an agentic AI assistant specialized in marketing mix optimization for a pharmaceutical company.
You have access to past marketing mix data as context.
When given a user query containing constraints, goals, budget, channel, or other inputs, analyze these carefully and provide optimized marketing tactic recommendations.
If the query is a greeting, casual conversation, or unrelated to the context, respond appropriately and naturally.
Use the provided context only when it is relevant to the query.
If the context is insufficient to answer the query confidently, say so and ask the user a clarification question.
Keep your response concise and clear. For questions related to planning, explain your reasoning in your response.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

def retrieve_similar_documents(query, top_k=5):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k=top_k)
    docs = [Document(page_content=df.iloc[idx]["text"]) for idx in indices[0]]
    return docs

def generate_answer(user_query, context_documents, user_role):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format_prompt(
        user_query=user_query,
        document_context=context_text,
        user_role=user_role
    )
    # Use invoke method instead of __call__ to avoid deprecation warning and error
    response_chain = LANGUAGE_MODEL.invoke(formatted_prompt.to_string())
    print(response_chain)
    ans_split = response_chain.split("</think>")
    print(ans_split)
    return ans_split[1].strip() 

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
    docs = retrieve_similar_documents(query)
    if not docs:
        answer = "Not enough relevant data found. Please rephrase or provide more context."
    else:
        print("generating answer")
        answer = generate_answer(query, docs, session["role"])
        print("answer generated")
        # Ensure answer is a string for template rendering
        if not isinstance(answer, str):
            answer = str(answer)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        # Return JSON with answer only for AJAX requests
        return jsonify({"answer": answer})
    else:
        return render_template("chat.html", username=session["username"], role=session["role"], query=query, answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
