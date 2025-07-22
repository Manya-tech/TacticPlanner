from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import faiss
import uvicorn
import jinja2

# ---------------------- App Setup ----------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------- In-Memory Store ----------------------
current_user = {"username": None, "role": None}

# ---------------------- Load Models and Data ----------------------
df = pd.read_excel("department_marketing_mix_data.xlsx")
df["text"] = df.apply(lambda row: f"{row['Department']} department ran {row['Channel']} campaign in {row['Year']}. Spend: {row['Spend']}, ROI: {row['ROI']}, Incremental ROI: {row['Incremental ROI']}", axis=1)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_mmm_index.index")

LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

PROMPT_TEMPLATE = """
You are a strategic {user_role} assistant. Use the provided context to help with marketing tactic planning.
If the context is insufficient to answer the query confidently, say so and ask the user a clarification question.
Limit your response to 3 sentences max. Be concise and data-driven.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# ---------------------- Helper Functions ----------------------
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
    # response = LANGUAGE_MODEL(formatted_prompt.to_string())
    # return response
    response_chain = formatted_prompt | LANGUAGE_MODEL
    return response_chain.invoke()

# ---------------------- Routes ----------------------

@app.get("/", response_class=HTMLResponse)
async def signup_get(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup_post(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    # Store info in memory
    current_user["username"] = username
    current_user["role"] = role
    return RedirectResponse(url="/chat", status_code=303)

@app.get("/chat", response_class=HTMLResponse)
async def chat_get(request: Request):
    if not current_user["username"] or not current_user["role"]:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": current_user["username"],
        "role": current_user["role"]
    })

@app.post("/chat")
async def chat_post(request: Request, query: str = Form(...)):
    if not current_user["username"] or not current_user["role"]:
        return RedirectResponse(url="/", status_code=303)
    
    docs = retrieve_similar_documents(query)
    if not docs:
        answer = "Not enough relevant data found. Please rephrase or provide more context."
    else:
        answer = generate_answer(query, docs, current_user["role"])

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "username": current_user["username"],
        "role": current_user["role"],
        "query": query,
        "answer": answer
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)