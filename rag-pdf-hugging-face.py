from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "REPLACE_WITH_API_KEY"
# Load the PDF file
pdf_loader = PyPDFLoader("492301.pdf")
documents = pdf_loader.load()

print(f"Loaded {len(documents)} pages from the PDF.")

# Split the document into chunks of 1000 characters with 200 overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")

# Initialize the Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index from the document chunks
faiss_index = FAISS.from_documents(chunks, embedding_model)

print("FAISS index created.")

# Initialize the Hugging Face LLM (FLAN-T5 model)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 1, "max_length": 512}
)

# Create a retriever from the FAISS index
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Build the RAG model using the retriever and the LLM
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    chain_type="stuff", 
    return_source_documents=True
)
query = "what is amount of my electricity expense"
response = rag_chain({"query": query})

# Extract answer and sources
answer = response["result"]
source_docs = response["source_documents"]

print("Answer:", answer)
# print("\nSources:")
# for doc in source_docs:
#     print(doc.metadata["source"], "-", doc.page_content[:200], "...")
