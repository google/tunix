# Make sure you've installed the SDK:
# pip install google-generativeai
# export GEMINI_API_KEY=""

import os
import google.genai as genai

client = genai.Client(
    api_key="",
)

response = client.models.generate_content(
    model='gemini-2.5-flash', contents='Why is the sky blue?'
)
print(response.text)