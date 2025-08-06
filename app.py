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
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Add nl2br filter to Jinja2 environment using flask.Markup
app.jinja_env.filters['nl2br'] = lambda value: Markup(value.replace('\n', '<br>'))

# ---------------------- Load Models and Data ----------------------
df = pd.read_excel("data/Sample data.xlsx")
df["text"] = df.apply(lambda row: f"{row['Brand']} brand ran {row['Category']} campaign using {row['Tactic']} tactic in {row['Timeperiod']}. Spend: {row['$ Spend (MM)']}, Contribution: {row['$ Contribution']}, ROI: {row['ROI']}, Incremental ROI: {row['iROI']}", axis=1)

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
You are an AI assistant specialized in marketing mix optimization for a pharmaceutical company.
Your tasks are:
1.Analyze user inputs: goals, budgets, constraints, categories, and tactics.
2.Use historical data to recommend optimized tactic-level investments.
3.Ensure recommendations meet all constraints. If any data is missing, extrapolate reasonably and note assumptions.
4.Provide concise, correct, and actionable outputs with clear reasoning for planning-related queries.
5.For casual messages (e.g., "hello"), respond naturally and briefly.
6.Ask for clarification if input context is insufficient.
7.Ensure that your mathematical calculations are correct.

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

def retrieve_similar_documents(query, top_k=10):
    """
    Retrieves relevant documents from the dataframe based on filters extracted
    from the user query, and augments with a semantic search.
    """
    # --- 1. Extract entities from the query ---
    
    # Years: Find all years mentioned, and identify which ones have data.
    years_in_query = [int(y) for y in re.findall(r'(20\d{2})', query)]
    available_years = df['Timeperiod'].unique()
    years_with_data = [y for y in years_in_query if y in available_years]

    # Brands: Find all known brands mentioned.
    all_brands = df['Brand'].unique()
    brand_pattern = r'\b(' + '|'.join([re.escape(b) for b in all_brands]) + r')\b'
    mentioned_brands = re.findall(brand_pattern, query, re.IGNORECASE)

    # Categories: Find all known categories mentioned.
    all_categories = df['Category'].unique()
    category_pattern = r'\b(' + '|'.join([re.escape(c) for c in all_categories]) + r')\b'
    mentioned_categories = re.findall(category_pattern, query, re.IGNORECASE)

    # Tactics: Find all known tactics mentioned.
    all_tactics = df['Tactic'].unique()
    tactic_pattern = r'\b(' + '|'.join([re.escape(t) for t in all_tactics]) + r')\b'
    mentioned_tactics = re.findall(tactic_pattern, query, re.IGNORECASE)

    # --- 2. Filter dataframe based on extracted entities ---
    
    filtered_df = df.copy()

    # Apply filters only if the user mentioned corresponding entities.
    # This handles cases where the user wants "all brands" or "all categories".
    if years_with_data:
        filtered_df = filtered_df[filtered_df['Timeperiod'].isin(years_with_data)]
    
    if mentioned_brands:
        # Find the original case of the brand name for accurate filtering
        brands_to_filter = [b for b in all_brands if b.lower() in [mb.lower() for mb in mentioned_brands]]
        filtered_df = filtered_df[filtered_df['Brand'].isin(brands_to_filter)]
        
    if mentioned_categories:
        # Find the original case of the category name
        categories_to_filter = [c for c in all_categories if c.lower() in [mc.lower() for mc in mentioned_categories]]
        filtered_df = filtered_df[filtered_df['Category'].isin(categories_to_filter)]
        
    if mentioned_tactics:
        # Find the original case of the tactic name
        tactics_to_filter = [t for t in all_tactics if t.lower() in [mt.lower() for mt in mentioned_tactics]]
        filtered_df = filtered_df[filtered_df['Tactic'].isin(tactics_to_filter)]

    # --- 3. Handle cases with no direct filter matches (e.g., future year queries) ---
    
    # If filtering resulted in an empty dataframe, it might be a query about the future.
    # In this case, we can provide data from the most recent year available for the other filters.
    if filtered_df.empty and years_in_query and not years_with_data:
        temp_df = df.copy()
        if mentioned_brands:
            brands_to_filter = [b for b in all_brands if b.lower() in [mb.lower() for mb in mentioned_brands]]
            temp_df = temp_df[temp_df['Brand'].isin(brands_to_filter)]
        if mentioned_categories:
            categories_to_filter = [c for c in all_categories if c.lower() in [mc.lower() for mc in mentioned_categories]]
            temp_df = temp_df[temp_df['Category'].isin(categories_to_filter)]
        if mentioned_tactics:
            tactics_to_filter = [t for t in all_tactics if t.lower() in [mt.lower() for mt in mentioned_tactics]]
            temp_df = temp_df[temp_df['Tactic'].isin(tactics_to_filter)]
        
        if not temp_df.empty:
            latest_year = temp_df['Timeperiod'].max()
            filtered_df = temp_df[temp_df['Timeperiod'] == latest_year]
    
    # Sort by ROI if query is about optimization
    if re.search(r'budget|roi|spend|invest|objective|optimal', query, re.IGNORECASE):
        filtered_df = filtered_df.sort_values(by=['ROI', '$ Spend (MM)'], ascending=False)

    # --- 4. Combine filtered results with semantic search results ---
    
    # Convert filtered rows to Document objects
    docs = [Document(page_content=row['text']) for _, row in filtered_df.iterrows()]

    # Augment with semantic search results from FAISS
    query_embedding_raw = embed_model.embed_query(query)
    query_embedding = np.array(query_embedding_raw) / norm(query_embedding_raw)
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
    faiss_docs = [Document(page_content=df.iloc[idx]["text"]) for idx in I[0]]

    # Combine filtered and FAISS results, giving priority to filtered docs and removing duplicates.
    combined_docs = docs
    existing_texts = {doc.page_content for doc in docs}
    for doc in faiss_docs:
        if doc.page_content not in existing_texts:
            combined_docs.append(doc)
            existing_texts.add(doc.page_content)

    if not combined_docs:
            return [Document(page_content="I could not find any relevant historical data for your query. Please try rephrasing or check if you are using recognized brand, category, and tactic names.")]

    return combined_docs

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
        answer = ans_split[1].strip()
    else:
        answer = response_content.strip()

    # Add "Did I answer your question properly?" unless the query is a greeting
    greetings = ["hi", "hello", "how are you"]
    if not any(greeting in user_query.lower() for greeting in greetings):
        answer += "\n\nDo you have any follow up questions?"

    return answer

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

    # Ensure conversation history is initialized
    if "history" not in session:
        session["history"] = []

    # Append user query to history
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

    # Append LLM response to history
    session["history"].append(f"Agent: {answer}")

    # Format answer for display
    try:
        answer = format_answer_for_display(answer)
    except Exception as e:
        print(f"Formatting error: {e}")

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"answer": answer})
    else:
        return render_template("chat.html", username=session["username"], role=session["role"], query=query, answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)