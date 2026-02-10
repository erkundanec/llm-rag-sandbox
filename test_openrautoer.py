import requests
import os
import sys

# Try to load .env if present (prefer python-dotenv, fallback to manual parsing)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)

# Load API key from environment variable
API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
if not API_KEY or API_KEY == "your-api-key-here":
    print("Error: OPENROUTER_API_KEY not set. Export it and retry.")
    sys.exit(1)

# OpenRouter public API base - FIXED: removed /chat/completions from base
BASE_URL = "https://openrouter.ai/api/v1"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _extract_reply(result):
    try:
        # OpenAI-like structure
        choices = result.get("choices", [])
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                # message.content (chat) or text (completion)
                msg = first.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    return msg.get("content")
                if first.get("text"):
                    return first.get("text")
        # fallbacks
        if isinstance(result.get("output"), str):
            return result.get("output")
    except Exception:
        pass
    return ""


def test_chat_completion():
    url = f"{BASE_URL}/chat/completions"
    print("Type your message (or 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": user_input}],
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            reply = _extract_reply(result)
            if reply:
                print("Assistant:", reply)
            else:
                print("Assistant (raw):", result)
        except requests.exceptions.ConnectionError:
            print(
                "Error: Could not connect to OpenRouter API. Check network and endpoint:",
                url,
            )
        except requests.exceptions.HTTPError as he:
            try:
                err = response.json()
            except Exception:
                err = response.text
            print("HTTP error:", he, err)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    test_chat_completion()