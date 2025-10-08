import streamlit as st
from huggingface_hub import InferenceClient
from datetime import datetime
import textwrap

# ----------------- Page Config -----------------
st.set_page_config(page_title="🤖 ChatBot with Hugging Face", layout="wide")
st.title("🤖 ChatBot with Hugging Face")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    HF_Token = st.text_input("Enter your Hugging Face Token:", type="password")
    model_choice = st.selectbox(
        "Choose a model:",
        [
            "mistralai/Mistral-7B-Instruct-v0.3",  # your main backend model
            "gpt2"  # fallback model
        ]
    )
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant.", "time": datetime.now().strftime("%H:%M:%S")}
        ]
        st.experimental_rerun()

# ----------------- Initialize Chat History -----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant.", "time": datetime.now().strftime("%H:%M:%S")}
    ]

# ----------------- Chat Input -----------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Enter your message:")
    submitted = st.form_submit_button("Send")

# ----------------- Process Input -----------------
if submitted and user_input and HF_Token:
    try:
        # Add user message once
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        # Call Hugging Face model
        with st.spinner("🤖 Typing..."):
            client = InferenceClient(model=model_choice, token=HF_Token)
            response = client.chat_completion(messages=st.session_state["messages"])

        reply = textwrap.fill(response.choices[0].message.content.strip(), width=80)

        # Add assistant reply once
        st.session_state["messages"].append({
            "role": "assistant",
            "content": reply,
            "time": datetime.now().strftime("%H:%M:%S")
        })

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ----------------- Display Chat -----------------
for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        continue
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    st.markdown(f"{avatar} **{msg['role'].capitalize()} ({msg['time']})**: {msg['content']}")

# ----------------- Export Chat -----------------
chat_text = "\n".join([
    f"{msg['role'].upper()} ({msg['time']}): {msg['content']}"
    for msg in st.session_state["messages"] if msg["role"] != "system"
])
st.download_button("📄 Export Chat", chat_text, file_name="chat_history.txt")
