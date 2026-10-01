from dotenv import load_dotenv
load_dotenv()  # read OLLAMA_HOST etc. from .env before ollama is imported

import ollama
import os

model = "llama3.2"

#Path to the input and output files
input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

#check if the input file exists
if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist.")
    exit(1)
# Read the grocery list from the input file
with open(input_file, 'r') as file:
    grocery_list = file.read().strip()

prompt = f"""
1. You are a grocery categorizer. Categorize the following grocery items into their respective categories.
2. Sort the items alphabetically within each category.
3. Present the categorized items in a structured format using bullet points and numbering.

Grocery items:
{grocery_list}
"""

#Send the prompt and get the response
try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print(generated_text)
    # Write the categorized grocery list to the output file
    with open(output_file, 'w') as file:
        file.write(generated_text)
    print(f"Categorized grocery list written to {output_file}")
except Exception as e:
    print(f"An error occurred while generating the response: {e}")
    exit(1)