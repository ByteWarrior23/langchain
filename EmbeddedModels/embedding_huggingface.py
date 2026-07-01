import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")},
)

text = "what is number of union territories in India, also name them"
vector = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Embedding dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

output = f"Text: {text}\nEmbedding dimension: {len(vector)}\nFirst 5 values: {vector[:5]}"
with open("output_embedding.txt", "w", encoding="utf-8") as f:
    f.write(output)
