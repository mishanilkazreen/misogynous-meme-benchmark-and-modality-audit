"""Quick sanity-check: send a text prompt to Gemini and print the response."""

import os

import google.generativeai as genai  # type: ignore[import-untyped,import-not-found]

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.1-flash-lite", safety_settings=safety_settings)

response = model.generate_content("What is 2+2?")
print(response.text)
