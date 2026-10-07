"""Any OpenAI client works; only base_url changes."""
from openai import OpenAI  # pip install openai

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="planner",
    messages=[{"role": "user", "content": "Plan a 3-step RAG pipeline."}],
)
print(r.choices[0].message.content)
