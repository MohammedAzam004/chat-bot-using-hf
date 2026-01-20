from huggingface_hub import InferenceClient
from getpass import getpass
import textwrap

MODEL_OPTIONS = [
    ("Zephyr 7B Beta (HuggingFaceH4)", "HuggingFaceH4/zephyr-7b-beta"),
]

def choose_model() -> str:
    print("Available models:")
    for idx, (label, _) in enumerate(MODEL_OPTIONS, start=1):
        print(f"{idx}. {label}")
    choice = input("Select model [1]: ").strip()
    if not choice:
        return MODEL_OPTIONS[0][1]
    index = int(choice) - 1
    if 0 <= index < len(MODEL_OPTIONS):
        return MODEL_OPTIONS[index][1]
    raise ValueError("Invalid selection.")

def build_prompt(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def run_inference(model_name: str, token: str, messages: list[dict]) -> str:
    client = InferenceClient(model=model_name, token=token)
    if model_name == "google/flan-t5-base":
        prompt = build_prompt(messages)
        return client.text_generation(prompt, max_new_tokens=256).strip()
    completion = client.chat_completion(messages=messages)
    return completion.choices[0].message.content.strip()

HF_Token = getpass("Enter token: ")
model_name = choose_model()
user_input = input("Enter input: ")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_input}
]

client_reply = run_inference(model_name, HF_Token, messages)
print(textwrap.fill(client_reply, width=80))