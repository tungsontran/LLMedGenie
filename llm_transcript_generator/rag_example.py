import bs4

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llm_transcript_generator.utils import *


loader = WebBaseLoader(
    web_paths=("https://d2l.ai/chapter_linear-regression/linear-regression.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("page-content")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Load tokens from file
token_file_path = "api_secret.txt"
tokens = load_tokens(token_file_path)
set_environment_variables(tokens)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=splits, embedding=hf)

retriever = vectorstore.as_retriever()
