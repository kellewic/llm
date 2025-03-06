import glob, os, sys

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from dotenv import find_dotenv, load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from sklearn.manifold import TSNE

load_dotenv(dotenv_path=find_dotenv())

openai_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_MODEL')
db_name = 'vector_db'


## Read in the entire knowledge base
def add_doc_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    documents.extend([add_doc_metadata(doc, doc_type) for doc in folder_docs])


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)

print(f"Chunks created: {len(chunks)}")
print(f"Document types found: {', '.join(doc_types)}")

embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x = reduced_vectors[:, 0],
    y = reduced_vectors[:, 1],
    mode = 'markers',
    marker = dict(size=5, color=colors, opacity=0.8),
    text = [f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo = 'text'
)])

fig.update_layout(
    title = '2D Chroma Vector Store Visualization',
    scene = dict(xaxis_title='x',yaxis_title='y'),
    width = 800,
    height = 600,
    margin = dict(r=20, b=10, l=10, t=40)
)

fig.write_html('vectors.html')

llm = ChatOpenAI(temperature=0.7, model_name=openai_model)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory,
    callbacks = [StdOutCallbackHandler()])

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(fn=chat, type="messages").launch(share=True)

