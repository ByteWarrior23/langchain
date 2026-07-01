import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512, "return_full_text": False},
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is number of union territories in India, also name them")

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(result.content)

print(result.content)
