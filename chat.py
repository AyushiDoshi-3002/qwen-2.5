import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 3B is too large for Streamlit Cloud free tier RAM

st.set_page_config(page_title="Qwen2.5 Chat", page_icon="💬")
st.title("💬 Qwen2.5 Chat")


@st.cache_resource(show_spinner="Loading model... this can take a minute on first run")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
