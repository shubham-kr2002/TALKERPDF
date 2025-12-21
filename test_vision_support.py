#!/usr/bin/env python3
"""Test if Llama-4-Scout supports vision on Groq"""

import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Create a 100x100 white image (valid size for vision API)
img = Image.new('RGB', (100, 100), color='white')
buffer = BytesIO()
img.save(buffer, format='JPEG', quality=60)
buffer.seek(0)
test_image_b64 = base64.b64encode(buffer.read()).decode('utf-8')

print("Testing Llama-4-Scout with vision...")
try:
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image briefly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{test_image_b64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=50
    )
    print(f"✓ Success! Response: {response.choices[0].message.content}")
    print("\n✅ Llama-4-Scout DOES support vision!")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\n⚠ Llama-4-Scout vision API issue")
    print("\nAlternative: Use OpenAI GPT-4 Vision or disable vision features temporarily")
