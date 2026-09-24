import uuid

import gradio as gr
import httpx

API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REQUEST_ID = "req-12345"
DEFAULT_USER_ID = "005xx000001SvU"


async def _ingest_docs(force_reembed: bool) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{API_BASE_URL}/api/v1/ingest", json={"force_reembed": force_reembed})
            response.raise_for_status()
            payload = response.json()
            return (
                "Ingestion successful. "
                f"Loaded {payload['documents_loaded']} docs, indexed {payload['chunks_indexed']} chunks, "
                f"reused_existing={payload['reused_existing']}."
            )
    except Exception as exc:
        return f"Ingestion failed: {exc}"


async def ingest_docs() -> str:
    return await _ingest_docs(force_reembed=False)


async def force_reembed_docs() -> str:
    return await _ingest_docs(force_reembed=True)


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


async def ask_question(
    query: str,
    history: list[dict[str, str]] | None,
    session_id: str,
    request_id: str,
    user_id: str,
) -> tuple[list[dict[str, str]], str, str]:
    messages = list(history or [])
    if not query.strip():
        return messages, query, "Please enter a query."

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/query",
                json={
                    "requestId": request_id or DEFAULT_REQUEST_ID,
                    "userId": user_id or DEFAULT_USER_ID,
                    "sessionId": session_id,
                    "query": query,
                },
            )
            response.raise_for_status()
            payload = response.json()

        response_payload = payload.get("response", {})
        answer = response_payload.get("answer", "")
        sources_text = "\n\n".join(
            f"Source: {item['source']}\nSnippet: {item['snippet']}" for item in response_payload.get("sources", [])
        )
        standalone = response_payload.get("standaloneQuery", "")
        if standalone and standalone != query:
            sources_text = f"Standalone question: {standalone}\n\n{sources_text}"
    except Exception as exc:
        answer = f"Query failed: {exc}"
        sources_text = ""

    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": answer})
    return messages, "", sources_text


async def reset_chat(session_id: str) -> tuple[list[dict[str, str]], str, str]:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.delete(f"{API_BASE_URL}/api/v1/session/{session_id}")
    except Exception:
        pass
    return [], new_session_id(), ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Brain Box RAG Tester") as demo:
        gr.Markdown("# Brain Box RAG Tester")
        gr.Markdown("Use this UI to trigger ingestion and test retrieval through the FastAPI backend.")

        with gr.Row():
            ingest_button = gr.Button("Ingest Confluence Docs", variant="primary")
            force_ingest_button = gr.Button("Force Ingest Docs", variant="stop")
            ingest_output = gr.Textbox(label="Ingestion Status")

        session_state = gr.State(value=new_session_id)
        chatbot = gr.Chatbot(label="Conversation", height=420)
        query_input = gr.Textbox(label="Question", lines=3, placeholder="Ask something from Confluence knowledge")
        with gr.Row():
            ask_button = gr.Button("Ask", variant="primary")
            new_chat_button = gr.Button("New Chat")
        request_id_input = gr.Textbox(label="Request ID", value=DEFAULT_REQUEST_ID)
        user_id_input = gr.Textbox(label="User ID", value=DEFAULT_USER_ID)
        sources_output = gr.Textbox(label="Retrieved Sources", lines=12)

        ingest_button.click(fn=ingest_docs, outputs=[ingest_output])
        force_ingest_button.click(fn=force_reembed_docs, outputs=[ingest_output])

        ask_inputs = [query_input, chatbot, session_state, request_id_input, user_id_input]
        ask_outputs = [chatbot, query_input, sources_output]
        ask_button.click(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)
        query_input.submit(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)
        new_chat_button.click(
            fn=reset_chat,
            inputs=[session_state],
            outputs=[chatbot, session_state, sources_output],
        )

    return demo


def launch() -> None:
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    launch()
