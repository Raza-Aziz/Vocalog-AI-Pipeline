from langgraph.graph import START, END, StateGraph
from state import MoMGraphState
from nodes import generate_markdown_mom, generate_mom

builder = StateGraph(MoMGraphState)

builder.add_node("generate_mom", generate_mom)
builder.add_node('generate_markdown_mom', generate_markdown_mom)

builder.add_edge(START, 'generate_mom')
builder.add_edge('generate_mom', 'generate_markdown_mom')
builder.add_edge('generate_markdown_mom', END)

mom_graph = builder.compile()