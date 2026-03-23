import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "qwen/qwen3-32b",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)
data = response.json()
print(data["choices"][0]["message"]["content"])