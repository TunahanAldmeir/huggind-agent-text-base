import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore
from transformers.agents import HfApiEngine, ReactJsonAgent


knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model)
vectordb.save_local("./trainedData")
print('data saved')
all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))
print(all_sources)
class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "source": {"type": "string", "description": ""},
        "number_of_documents": {
            "type": "string",
            "description": "the number of documents to retrieve. Stay under 10 to avoid drowning in docs",
        },
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, all_sources: str, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.inputs["source"]["description"] = (
            f"The source of the documents to search, as a str representation of a list. Possible values in the list are: {all_sources}. If this argument is not provided, all sources will be searched.".replace(
                "'", "`"
            )
        )

    def forward(self, query: str, source: str = None, number_of_documents=7) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        number_of_documents = int(number_of_documents)

        if source:
            if isinstance(source, str) and "[" not in str(source):  # if the source is not representing a list
                source = [source]
            source = json.loads(str(source).replace("'", '"'))

        docs = self.vectordb.similarity_search(
            query,
            filter=({"source": source} if source else None),
            k=number_of_documents,
        )

        if len(docs) == 0:
            return "No documents found with this filtering. Try removing the source filter."
        return "Retrieved documents:\n\n" + "\n===Document===\n".join([doc.page_content for doc in docs])
llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

retriever_tool = RetrieverTool(vectordb=vectordb, all_sources=all_sources)
agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, verbose=0,system_prompt="you are god of articles",)

agent_output = agent.run("write me article about what you know  it should be at least ten lines br creative base on your trained knowledge make it look better")

print("Final output:")
print(agent_output)
