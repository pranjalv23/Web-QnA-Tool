from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core.logging_helpers import logger
from qa_workflow.workflow import AgentState, agent, rewrite, generate, grade_documents, docs_splits
from database.database_manager import create_retriever


logger.info(" - Initializing")
workflow = StateGraph(AgentState)
# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retriever_tool = create_retriever(docs_splits)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

agent_workflow = workflow.compile()


def process():
    config = {"configurable": {"thread_id": "def234"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Professor X: Goodbye!")
            break

        inputs = {
            "messages": [
                ("user", user_input),
            ]
        }

        result = agent_workflow.invoke(inputs, config=config, stream_mode="values")
        logger.info(f"Professor X: {result['messages'][-1].content}")
        return result['messages'][-1].content