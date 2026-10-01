from dotenv import load_dotenv
load_dotenv()  # read OLLAMA_HOST etc. from .env before ollama is imported

import os
import requests
import json
from urllib.parse import urlsplit


def ollama_base_url(value):
    """Normalize OLLAMA_HOST the way the ollama client does: default scheme
    http and host 127.0.0.1; port 11434 when no scheme is given, otherwise the
    scheme's default (80 for http, 443 for https)."""
    value = (value or "").strip()
    explicit = "://" in value
    if not explicit:
        value = "http://" + value
    parts = urlsplit(value)
    scheme = parts.scheme or "http"
    host = parts.hostname or "127.0.0.1"
    default_port = {"http": 80, "https": 443}[scheme] if explicit else 11434
    port = parts.port or default_port
    if ":" in host:
        host = f"[{host}]"  # IPv6
    return f"{scheme}://{host}:{port}{parts.path.rstrip('/')}"


url = f"{ollama_base_url(os.getenv('OLLAMA_HOST'))}/api/generate"

data = {
    "model": "llama3.2",
    "prompt": "Tell me a joke?",
}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # Iterate over the response stream
    for line in response.iter_lines():
        if line:
            # Decode the line and print it
            decoded_line = line.decode('utf-8')
            result = json.loads(decoded_line)
            # Get the generated text from the result
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)
    