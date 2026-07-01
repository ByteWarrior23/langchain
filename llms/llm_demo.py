from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
)

result = llm.invoke("what is number of union territories in India")

print(result.content)