import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
)

result = model.invoke("what is number of union territories in India, also name them ")

print(result.content)
