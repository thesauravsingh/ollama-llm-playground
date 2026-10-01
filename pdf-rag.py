from dotenv import load_dotenv
load_dotenv()  # read OLLAMA_HOST etc. from .env before ollama is imported

## 1. ingest pdf files
# 2. extract text from pdf files and split into chunks
# 3. send the chunks to the embedding model
# 4. save the embeddings to a vector store
# 5. perform a similarity search on the vector database to find similar chunks
# 6. retrieve the similar documents and display them

## run the requirements.txt file to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
# We may use one of the following loaders


import nltk

# Download required NLTK models if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")



doc_path = "./data/ai_history_sample.pdf"
model = "llama3.2"


#load PDF file uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading....")
else:
    print("Upload a PDF file to continue...")

#preview the data
content = data[0].page_content
print(content[:1000])  # Display the first 1000 characters of the content

# ==== End of page ingestion ====

# ====  Extract text from PDF files and split into chunks ====
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

#Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Adjust chunk size as needed
    chunk_overlap=300,  # Adjust overlap as needed, greater overlap can help with context
)
chunks = text_splitter.split_documents(data)
print("done splitting....")

# print(f"Number of chunks created: {len(chunks)}")
# print(f"First chunk: {chunks[0]}")  # Display the first chunk

# ==== Add the chunks to vector db ====
import ollama
ollama.pull("nomic-embed-text")
# Initialize the Ollama embeddings model
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple_rag",
)
print("done embedding....")

## === Retrieval ====
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm  = ChatOllama(model=model, temperature=0.1)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your taske is to answer 5 different versions of the given user question
    to retrieve relevant information from a vector database. By generating multiple perspective on the user question, your goal is 
    to help user overcome some of the limitations of the distance based similarity search. Provide these alternative questions seperated 
    by newlines.  Original question is:
{question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT
)

# RAG Prompt
template = """Answer the questions based only on the following context. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# res = chain.invoke(input=("what is the document about?",))
res = chain.invoke(input=("what are the main observations?",))
print(res)
# ==== End of retrieval ====