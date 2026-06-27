import asyncio
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

load_dotenv()

st.title("🤖 LlamaIndex Autonomous Agentic RAG")
st.caption("Multi-document reasoning orchestrator using Gemini 2.5 Flash")

SYSTEM_PROMPT = """You are an autonomous technical analyst with access to two documentation tools.

RULES (follow strictly):
1. You MUST call the appropriate tool(s) before answering ANY question.
2. For questions about n8n, call `n8n_documentation_tool`.
3. For questions about Claude Code, call `claude_code_documentation_tool`.
4. For comparison questions, call BOTH tools.
5. Never answer from memory alone — always retrieve from the tools first.
6. If you have already used a tool and have the information, synthesize a final answer.
"""

@st.cache_resource
def configure_agent_brain():
    Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
    Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.1)

    # Load n8n vector index
    n8n_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="storage/n8n")
    )
    n8n_engine = n8n_index.as_query_engine(similarity_top_k=3)

    # Load Claude Code vector index
    claude_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="storage/claude")
    )
    claude_engine = claude_index.as_query_engine(similarity_top_k=3)

    tools = [
        QueryEngineTool(
            query_engine=n8n_engine,
            metadata=ToolMetadata(
                name="n8n_documentation_tool",
                description=(
                    "Use this tool to retrieve technical information about n8n, including "
                    "its core architecture, node types, AI agent capabilities, and deployment models."
                )
            )
        ),
        QueryEngineTool(
            query_engine=claude_engine,
            metadata=ToolMetadata(
                name="claude_code_documentation_tool",
                description=(
                    "Use this tool to retrieve technical information about Claude Code, including "
                    "its tool execution model, core capabilities, terminal commands, and agentic loops."
                )
            )
        )
    ]

    return ReActAgent(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
        context=SYSTEM_PROMPT,
        max_iterations=10,
    )

agent = configure_agent_brain()

# Initialize chat session state
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your autonomous technical analyst. I can cross-reference information about n8n and Claude Code. What are we analyzing today?"
        }
    ]

# Render chat history
for msg in st.session_state.agent_messages:
    avatar = "🧠" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

# Handle new user input
if user_input := st.chat_input("Ask a complex question cross-referencing your documents..."):
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    st.session_state.agent_messages.append({"role": "user", "content": user_input})

    # Build chat history from session state to pass real context to the agent
    # (excludes the initial greeting and the current user message already appended)
    chat_history = []
    for msg in st.session_state.agent_messages[1:-1]:  # skip greeting + current input
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        chat_history.append(ChatMessage(role=role, content=msg["content"]))

    # agent.run() must be called INSIDE the coroutine so asyncio.create_task() has a running loop
    async def _run_agent(query: str, history: list):
        return await agent.run(user_msg=query, chat_history=history)

    with st.spinner("Agent is reasoning and invoking documentation tools..."):
        response = asyncio.run(_run_agent(user_input, chat_history))
        # AgentOutput.response is a ChatMessage; .content is the final text
        ai_response = response.response.content

    with st.chat_message("assistant", avatar="🧠"):
        st.write(ai_response)
    st.session_state.agent_messages.append({"role": "assistant", "content": ai_response})
