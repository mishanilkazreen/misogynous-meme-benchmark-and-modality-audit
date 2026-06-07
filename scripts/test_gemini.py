"""Quick sanity-check: send a text prompt to Gemini and print the response."""

import os

import google.generativeai as genai  # type: ignore[import-untyped,import-not-found]

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.1-flash-lite")

response = model.generate_content("What is 2+2?")
print(response.text)
