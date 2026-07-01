import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

try:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    text = "what is number of union territories in India, also name them"
    vector = embeddings.embed_query(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

    output = f"Text: {text}\nEmbedding dimension: {len(vector)}\nFirst 5 values: {vector[:5]}"
    with open("output_embedding_openai.txt", "w", encoding="utf-8") as f:
        f.write(output)
except Exception as e:
    print(f"OpenAI embeddings error: {e}")
    print("Note: Your API key is an OpenRouter key which does not support embeddings.")
    print("Use HuggingFace embeddings instead.")