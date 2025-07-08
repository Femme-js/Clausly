from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

class VectorStoreIndexer:
    def __init__(self, config):
        self.embedding = HuggingFaceEmbeddings(model_name=config["embedding_model"])
        self.vectorstore = Cassandra(
            embedding=self.embedding,
            table_name=config["cass_table"],
            session=config["cass_session"],
            keyspace=config["cass_keyspace"],
        )

    def index_urls(self, urls):
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_flat = [doc for sublist in docs for doc in sublist]
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
        split_docs = splitter.split_documents(docs_flat)
        self.vectorstore.add_documents(split_docs)
        print(f"Inserted {len(split_docs)} documents.")

    def as_retriever(self):
        return self.vectorstore.as_retriever()