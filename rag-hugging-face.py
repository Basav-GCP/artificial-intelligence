from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import os


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "REPLACE_WITH_API_KEY"

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # You can replace it with any suitable model.
    model_kwargs={"temperature": 0.3, "max_length": 256}
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "LangChain helps build applications powered by large language models.",
    "FAISS is a tool for efficient vector search and similarity matching.",
    "Hugging Face offers models for both text generation and embeddings."
]

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=10)
docs = text_splitter.create_documents(documents)

# Create a FAISS index using the documents and embeddings model
faiss_index = FAISS.from_documents(docs, embedding_model)

# Create the retriever from the FAISS index
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k":2})

# Ensure the parameters match expected types and names
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type="stuff",  # Use "stuff" for basic concatenation of retrieved documents
    return_source_documents=True  # Optional, to return sources along with the answer
)

# query = "What does LangChain do?"
# response = rag_chain.run(query)

query = "What does LangChain do?"
response = rag_chain.apply([{"query": query}])

# Access the results
print("Answer:", response[0]["result"])
print("Source Documents:", response[0]["source_documents"])
