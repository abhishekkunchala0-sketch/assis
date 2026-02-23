"""Quick test script for OpenRouter API."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Test imports
print("Testing imports...")

try:
    from src.utils.config import Config
    from src.data.schema_loader import SchemaLoader
    from src.agent.sql_generator import create_llm
    print("[OK] Imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Load config
print("\nLoading config...")
config = Config("config.yaml")
print(f"LLM Provider: {config.llm_provider}")
print(f"LLM Model: {config.llm_model}")
print(f"Base URL: {config.llm_base_url}")

# Test API key (check various providers)
api_key = (
    os.environ.get("OPENROUTER_API_KEY") or 
    os.environ.get("OPENAI_API_KEY") or
    os.environ.get("HF_TOKEN") or
    os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
if api_key:
    print(f"[OK] API key found: {api_key[:10]}...")
else:
    print("[INFO] No API key found - will use free tier or mock")

# Create LLM
print("\nCreating LLM...")
try:
    llm = create_llm(
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=0.1,
        base_url=config.llm_base_url
    )
    print(f"[OK] LLM created: {llm}")
except Exception as e:
    print(f"[FAIL] LLM creation failed: {e}")
    sys.exit(1)

# Test LLM
print("\nTesting LLM with simple query...")
try:
    response = llm.invoke("What is 2+2? Answer briefly.")
    print(f"[OK] LLM response: {response}")
except Exception as e:
    print(f"[FAIL] LLM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed!")
