import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Annotated
import operator
from typing_extensions import TypedDict

from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq


#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


llm=ChatGroq(model="qwen-qwq-32b")

# --- Schema for Planning ---
class Section(BaseModel):
    name: str = Field(description="Title of this blog section")
    description: str = Field(description="What this section should discuss")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections for the blog")

# --- Planning LLM ---
planner = llm.with_structured_output(Sections)

# --- Shared LangGraph State ---
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_blog: str

# --- Worker State ---
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

# --- Orchestrator Node ---
def orchestrator(state: State):
    plan = planner.invoke([
        SystemMessage(content="Plan a detailed blog structure."),
        HumanMessage(content=f"The blog topic is: {state['topic']}")
    ])
    return {"sections": plan.sections}

# --- Worker Node ---
def llm_call(state: WorkerState):
    response = llm.invoke([
        SystemMessage(
            content="Write a detailed blog section in markdown using the given title and description. Do not include a preamble or greeting."
        ),
        HumanMessage(
            content=f"Title: {state['section'].name}\nDescription: {state['section'].description}"
        )
    ])
    return {"completed_sections": [response.content]}

# --- Worker Assignment ---
def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# --- Synthesizer ---
def synthesizer(state: State):
    full_blog = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_blog": full_blog}

# --- LangGraph ---
builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_call", llm_call)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", END)

blog_graph = builder.compile()

# --- Streamlit App ---
st.set_page_config(page_title="AI Blog Generator", layout="centered")

st.title("📝 AI Blog Generator with Gemini + LangGraph")
st.markdown("Generate a blog from just a topic using orchestrator-worker agents!")

topic = st.text_input("Enter your blog topic", value="The Role of Agentic AI in Personalized Learning")

if st.button("Generate Blog"):
    with st.spinner("Generating... please wait..."):
        final_state = blog_graph.invoke({"topic": topic})
        blog = final_state["final_blog"]
        st.markdown("### ✨ Generated Blog")
        st.markdown(blog, unsafe_allow_html=True)

        st.download_button("📥 Download as Markdown", data=blog, file_name="generated_blog.md")

        st.text_area("📋 Copy for LinkedIn", value=blog, height=400)
