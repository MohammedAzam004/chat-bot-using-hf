import streamlit as st
from huggingface_hub import InferenceClient
from datetime import datetime
import textwrap

MODEL_OPTIONS = [
    ("Zephyr 7B Beta (HuggingFaceH4)", "HuggingFaceH4/zephyr-7b-beta"),
   ]

st.set_page_config(page_title="🤖 ChatBot ", layout="wide")
st.title("🤖 ChatBot ")

def build_prompt(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def generate_response(model_name: str, messages: list[dict], token: str) -> str:
    client = InferenceClient(model=model_name, token=token)
    if model_name == "google/flan-t5-base":
        prompt = build_prompt(messages)
        return client.text_generation(prompt, max_new_tokens=256).strip()
    completion = client.chat_completion(messages=messages)
    return completion.choices[0].message.content.strip()

with st.sidebar:
    st.header("⚙️ Settings")
    HF_Token = st.text_input("Enter your Hugging Face Token:", type="password")
    model_label = st.selectbox("Choose a model:", [name for name, _ in MODEL_OPTIONS], index=0)
    model_name = dict(MODEL_OPTIONS)[model_label]
    st.caption(f"Model: {model_name}")
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant.", "time": datetime.now().strftime("%H:%M:%S")}
        ]
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant.", "time": datetime.now().strftime("%H:%M:%S")}
    ]

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Enter your message:")
    submitted = st.form_submit_button("Send")

if submitted and not HF_Token:
    st.warning("Please provide a Hugging Face token before sending messages.")

if submitted and user_input and HF_Token:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["messages"].append({"role": "user", "content": user_input, "time": timestamp})

    api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"]]

    try:
        with st.spinner("🤖 Typing..."):
            reply_raw = generate_response(model_name, api_messages, HF_Token)
    except Exception as err:
        st.session_state["messages"].pop()
        st.error(f"❌ Error: {err}")
        reply_raw = None

    if reply_raw:
        reply_text = textwrap.fill(reply_raw, width=80)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": reply_text,
            "time": datetime.now().strftime("%H:%M:%S")
        })

for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        continue
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    st.markdown(f"{avatar} **{msg['role'].capitalize()} ({msg['time']})**: {msg['content']}")

chat_text = "\n".join([
    f"{msg['role'].upper()} ({msg['time']}): {msg['content']}"
    for msg in st.session_state["messages"] if msg["role"] != "system"
])
st.download_button("📄 Export Chat", chat_text, file_name="chat_history.txt")