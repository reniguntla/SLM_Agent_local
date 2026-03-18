import json
from typing import Dict, List, Optional

import requests
import streamlit as st

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_LABEL = "Phi-3"
DEFAULT_MODEL_TAG = "phi3"
MAX_INTERACTIONS = 10
APPROX_CONTEXT_TOKEN_LIMIT = 3000
SUPPORTED_MODELS: Dict[str, str] = {
    "Phi-3": "phi3",
    "Gemma": "gemma",
    "Mistral": "mistral",
}
SYSTEM_PROMPT = (
    "You are a helpful assistant powered by a local Small Language Model. "
    "Answer only the user's asked question using the recent conversation for context when helpful. "
    "Be concise, clear, and avoid unrelated information."
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL_LABEL
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = ""


@st.cache_data(ttl=10)
def get_ollama_data(endpoint: str) -> Optional[dict]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None



def get_available_supported_models() -> Dict[str, str]:
    tags_response = get_ollama_data("/api/tags") or {}
    installed_models = {
        model_info.get("name", "").split(":")[0]
        for model_info in tags_response.get("models", [])
    }

    available = {
        label: tag
        for label, tag in SUPPORTED_MODELS.items()
        if tag.split(":")[0] in installed_models
    }
    return available or {DEFAULT_MODEL_LABEL: DEFAULT_MODEL_TAG}



def get_running_models() -> List[str]:
    ps_response = get_ollama_data("/api/ps") or {}
    return [
        model_info.get("name", "")
        for model_info in ps_response.get("models", [])
        if model_info.get("name")
    ]



def approximate_tokens(text: str) -> int:
    return max(1, len(text) // 4)



def trim_messages(messages: List[dict], max_interactions: int, max_tokens: int) -> List[dict]:
    trimmed = messages[-(max_interactions * 2) :]

    while trimmed:
        total_tokens = sum(approximate_tokens(message["content"]) for message in trimmed)
        if total_tokens <= max_tokens:
            break
        trimmed = trimmed[2:]

    return trimmed



def build_chat_messages(history: List[dict], prompt: str) -> List[dict]:
    trimmed_history = trim_messages(history, MAX_INTERACTIONS, APPROX_CONTEXT_TOKEN_LIMIT)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *trimmed_history,
        {"role": "user", "content": prompt},
    ]



def stream_ollama_response(model_tag: str, messages: List[dict]):
    payload = {"model": model_tag, "messages": messages, "stream": True}
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            message = chunk.get("message", {})
            content = message.get("content", "")
            if content:
                yield content



def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.prompt_input = ""



def render_sidebar(model_options: Dict[str, str]) -> str:
    st.sidebar.header("SLM Configuration")

    selected_model = st.sidebar.selectbox(
        "SLM Selection",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.selected_model)
        if st.session_state.selected_model in model_options
        else 0,
    )
    st.session_state.selected_model = selected_model

    selected_tag = model_options[selected_model]
    running_models = get_running_models()
    selected_is_running = any(name.split(":")[0] == selected_tag for name in running_models)
    installed_labels = ", ".join(model_options.keys())

    st.sidebar.markdown(f"**Model Status:** Currently Using {selected_model}")
    st.sidebar.caption(f"Installed supported models detected in Ollama: {installed_labels}")
    if selected_model == DEFAULT_MODEL_LABEL:
        if selected_is_running:
            st.sidebar.success("Phi-3 is currently running in local Ollama.")
        else:
            st.sidebar.warning(
                "Phi-3 is selected but not currently running. Ollama will try to start it locally on the first request."
            )

    if st.sidebar.button("Clear Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    return selected_tag



def render_chat_history() -> None:
    st.subheader("Conversation Window")
    if not st.session_state.messages:
        st.info("Start the conversation by asking a question below.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



def main() -> None:
    st.set_page_config(page_title="SLM Chat", page_icon="💬", layout="wide")
    init_session_state()

    st.title("Local SLM Chat Application")
    st.write(
        "Interact with locally hosted Small Language Models through Ollama. "
        f"The app keeps the last {MAX_INTERACTIONS} question-answer interactions for context."
    )

    model_options = get_available_supported_models()
    selected_model_tag = render_sidebar(model_options)

    if get_ollama_data("/api/tags") is None:
        st.error(
            "Ollama is not reachable at http://localhost:11434. Start Ollama locally and ensure Phi-3 is installed."
        )
        st.stop()

    st.markdown(f"**Currently Using Model:** {st.session_state.selected_model}")
    render_chat_history()

    with st.form("question_form", clear_on_submit=True):
        prompt = st.text_input("Question Input", placeholder="Type your question here")
        submit_col, clear_col = st.columns(2)
        with submit_col:
            submitted = st.form_submit_button("Submit", use_container_width=True)
        with clear_col:
            clear_requested = st.form_submit_button("Clear Conversation", use_container_width=True)

    if clear_requested:
        clear_conversation()
        st.rerun()

    if not submitted or not prompt.strip():
        return

    prompt = prompt.strip()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = trim_messages(
        st.session_state.messages, MAX_INTERACTIONS, APPROX_CONTEXT_TOKEN_LIMIT
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated_response = ""
        try:
            chat_messages = build_chat_messages(st.session_state.messages[:-1], prompt)
            for token in stream_ollama_response(selected_model_tag, chat_messages):
                accumulated_response += token
                placeholder.markdown(accumulated_response)
        except requests.RequestException as exc:
            accumulated_response = (
                "Unable to generate a response from Ollama. "
                "Please verify that the selected model is installed locally and Ollama is running. "
                f"Details: {exc}"
            )
            placeholder.error(accumulated_response)
        except json.JSONDecodeError:
            accumulated_response = "Received an invalid streaming response from Ollama."
            placeholder.error(accumulated_response)

    st.session_state.messages.append({"role": "assistant", "content": accumulated_response})
    st.session_state.messages = trim_messages(
        st.session_state.messages, MAX_INTERACTIONS, APPROX_CONTEXT_TOKEN_LIMIT
    )


if __name__ == "__main__":
    main()
