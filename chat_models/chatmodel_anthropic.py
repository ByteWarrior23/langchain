from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

try:
    model = ChatAnthropic(
        model="claude-3-5-haiku-20241022",
        temperature=0.5,
        max_tokens=200,
    )

    result = model.invoke("what is number of union territories in India, also name them ")
    print(result.content)
except Exception as e:
    print(f"Anthropic API error: {e}")
    print("Tip: Your Anthropic account may need credits. Use OpenRouter or another provider instead.")
