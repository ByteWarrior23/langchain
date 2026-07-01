from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-5",
    temperature=0.5,
    max_tokens=200,
)

result = model.invoke("what is number of union territories in India, also name them ")

print(result.content)
