import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Anthropic does not provide an embeddings API.")
print("They only offer chat/completions endpoints.")
print("Use HuggingFace embeddings instead.")
