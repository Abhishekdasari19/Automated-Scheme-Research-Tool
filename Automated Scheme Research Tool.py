#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from PyPDF2 import PdfReader
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# In[2]:


def fetch_pdf_content(pdf_url):
    try:
        response = requests.get(pdf_url)  # Fetch PDF from URL
        with open("temp.pdf", "wb") as f:
            f.write(response.content)  # Save the PDF locally
        
        reader = PdfReader("temp.pdf")  # Load the PDF
        content = ""
        for page in reader.pages:
            content += page.extract_text()  # Extract text from all pages
        
        return content
    except Exception as e:
        print(f"Error fetching content from {pdf_url}: {e}")
        return None

print("PDF fetching function ready!")


# In[3]:


def create_faiss_index(articles):
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(articles)
    index = faiss.IndexFlatL2(len(embeddings[0]))  
    index.add(embeddings) 
    return index, embeddings

print("FAISS index creation function ready!")


# In[4]:


def save_faiss_index(index, embeddings, file_name="faiss_store.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump({"index": index, "embeddings": embeddings}, f)
    print(f"FAISS index and embeddings saved to {file_name}!")

def load_faiss_index(file_name="faiss_store.pkl"):
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["embeddings"]

print("Save and load functions ready!")


# In[5]:


def get_answer(question, articles, index, model):
    query_embedding = model.encode([question])  
    distances, indices = index.search(query_embedding, k=1)
    answer_index = indices[0][0]  
    return articles[answer_index]

print("Question-answering system ready!")


# In[6]:


urls = [
    "https://mohua.gov.in/upload/uploadfiles/files/PMSVANidhi%20Guideline_English.pdf"
]

articles = [fetch_pdf_content(url) for url in urls if url]

if articles and articles[0]:
    print("Sample Article Content:\n", articles[0][:500]) 

    print("Creating FAISS index...")
    faiss_index, embeddings = create_faiss_index(articles)
    save_faiss_index(faiss_index, embeddings) 
    print("FAISS index created and saved!")
else:
    print("No valid articles fetched. Check your URLs.")


# In[8]:


faiss_index, embeddings = load_faiss_index()

model = SentenceTransformer('all-MiniLM-L6-v2')

question = "What are the Scheme Application Process of PMSVANidhi?"

if articles and articles[0]:
    answer = get_answer(question, articles, faiss_index, model)
    print("\nQuestion:", question)
    print("Answer:", answer)
else:
    print("No articles available for question answering.")


# In[ ]:




