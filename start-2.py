from dotenv import load_dotenv
load_dotenv()  # read OLLAMA_HOST etc. from .env before ollama is imported

import ollama

response = ollama.list()
#print(response)

# === Using ollama.chat to generate a response ===
# Ensure you have the ollama package installed: pip install ollama
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Tell me a joke?"},
    ],
)

# print(res["message"]["content"])

# == Using ollama.chat with stream=True to get a streaming response ==
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Tell me a joke?"},
    ],
    stream=True,
)
# Print the response as it streams 
#for chunk in res:
    # print(chunk["message"]["content"], end="", flush=True)

#show


modelfile = """
FROM llama3.2
SYSTEM You are Yoda Sensei. You speak like Yoda.
PARAMETER temperature 0.7
"""

ollama.create(model='Yoda', from_='llama3.2', system="Speak like Yoda, give spiritual advice", parameters={"temperature": 0.7})

# Now use the model
res = ollama.generate(model="yoda", prompt="Why do we fall?")
print(res["response"])

#delete the model
ollama.delete(model="yoda")