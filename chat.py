import streamlit as st
from huggingface_hub import InferenceClient

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."

st.set_page_config(page_title="Qwen2.5 Chat", page_icon="💬")
st.title("💬 Qwen2.5 Chat")

with st.sidebar:
    st.caption(f"Model: `{MODEL_NAME}`")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner=False)
def get_client() -> InferenceClient:
    # HF_TOKEN is optional but strongly recommended to avoid rate limits.
    # Add it to your app's Secrets: Settings → Secrets → HF_TOKEN = hf_xxx...
    token = st.secrets.get("HF_TOKEN", None)
    if not token:
        st.sidebar.warning(
            "⚠️ No HF_TOKEN set — unauthenticated requests are heavily rate-limited. "
            "Add your token in Settings → Secrets.",
            icon="🔑",
        )
    return InferenceClient(model=MODEL_NAME, token=token)


client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Prepend system prompt without storing it in session_state
        messages_for_api = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *st.session_state.messages,
        ]
        try:
            stream = client.chat_completion(
                messages=messages_for_api,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stream=True,
            )

            def token_stream():
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            response = st.write_stream(token_stream())

        except Exception as e:
            st.error(f"⚠️ Generation failed: {e}")
            response = None

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
