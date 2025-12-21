#!/usr/bin/env python3
"""Quick test to verify Groq vision API configuration"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Test 1: List available models
print("=" * 60)
print("TEST 1: Checking available Groq models...")
print("=" * 60)
try:
    models = groq_client.models.list()
    print(f"✓ Found {len(models.data)} models:")
    
    vision_models = []
    for model in models.data:
        model_id = model.id
        print(f"  - {model_id}")
        if 'vision' in model_id.lower() or 'llama-3.2' in model_id.lower():
            vision_models.append(model_id)
    
    if vision_models:
        print(f"\n✓ Vision-capable models found:")
        for vm in vision_models:
            print(f"  → {vm}")
    else:
        print("\n⚠ No vision models found. Available models:")
        print("  Try: llama-3.2-90b-vision-preview or llama-3.2-11b-vision-preview")
        
except Exception as e:
    print(f"✗ Error listing models: {e}")
    print("\nℹ Common Groq vision models (as of Dec 2024):")
    print("  - llama-3.2-90b-vision-preview")
    print("  - llama-3.2-11b-vision-preview")

# Test 2: Try a simple text completion to verify API key
print("\n" + "=" * 60)
print("TEST 2: Verifying API key with text completion...")
print("=" * 60)
try:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'API working!' in 2 words"}],
        max_tokens=10
    )
    print(f"✓ API Response: {response.choices[0].message.content}")
    print("✓ Groq API key is valid!")
except Exception as e:
    print(f"✗ API Error: {e}")
    print("\nℹ Please check:")
    print("  1. GROQ_API_KEY is set in .env file")
    print("  2. API key is valid (get one at https://console.groq.com/)")

print("\n" + "=" * 60)
print("RECOMMENDED ACTION:")
print("=" * 60)
print("Update core/ingestion.py line 177 with the correct vision model:")
print("  model=\"llama-3.2-90b-vision-preview\"  # or llama-3.2-11b-vision-preview")
print("=" * 60)
