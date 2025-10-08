# backend.py

from huggingface_hub import InferenceClient
from getpass import getpass
import textwrap

# Ask the user to enter their Hugging Face token securely
HF_Token = getpass("Enter token..")

# Create an InferenceClient object to interact with the chosen model
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=HF_Token)

# Take user input for the chat prompt
user_input = input("Enter input..")

# Define the conversation messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_input}
]

# Generate a chat-based response
response = client.chat_completion(messages=messages)

# Print the model's response nicely
print(textwrap.fill(response.choices[0].message.content.strip(), width=80))
