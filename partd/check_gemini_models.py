import os
import google.generativeai as genai

# Make sure your GEMINI_API_KEY is set
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GEMINI_API_KEY environment variable first!")

# Configure Gemini API
genai.configure(api_key=api_key)

# List available models
models = genai.list_models()
print("Available Gemini models:")
for m in models:
    # Use attribute access instead of dict access
    print(f"- {m.name}: supported methods = {getattr(m, 'supported_methods', [])}")

