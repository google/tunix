# Make sure you've installed the SDK:
# pip install google-generativeai
# export GEMINI_API_KEY="AIzaSyDw4scuSWz6akCjroAv69SSfsfEsUNbtuo"

import os
import google.genai as genai

client = genai.Client(
    api_key="AIzaSyDw4scuSWz6akCjroAv69SSfsfEsUNbtuo",
)

response = client.models.generate_content(
    model='gemini-2.5-flash', contents='Why is the sky blue?'
)
print(response.text)