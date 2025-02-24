from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_retriever(docs_splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(
        documents=docs_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="Professor X",
        description="This will retrieve the information about the urls provided by the user"
    )

    return retriever_tool