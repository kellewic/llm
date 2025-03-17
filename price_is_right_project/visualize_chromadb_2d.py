import pathlib
import chromadb
import numpy as np
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm

DB = "products_vectorstore"
MAXIMUM_DATAPOINTS = 30_000
collection_name = "products"

client = chromadb.PersistentClient(path=DB)
collection = client.get_collection(collection_name)

CATEGORIES = [
    'Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments',
    'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games'
]
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
vectors = np.array(result['embeddings'])
documents = result['documents']
categories = [metadata['category'] for metadata in result['metadatas']]
colors = [COLORS[CATEGORIES.index(c)] for c in categories]

tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=3, color=colors, opacity=0.7),
)])

fig.update_layout(
    title='2D Chroma Vectorstore Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y'),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40),
)

fig.write_html(f'{pathlib.PurePath(__file__).stem}.html')

