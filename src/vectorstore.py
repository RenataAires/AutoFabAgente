import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def criar_ou_carregar_vectorstore(documentos, diretorio_persistencia="./chroma_db"):
    """
    Cria ou carrega o banco vetorial Chroma usando embeddings gratuitos do HuggingFace.
    """
    # Modelo de embeddings leve e eficiente que roda localmente no CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(diretorio_persistencia) and os.listdir(diretorio_persistencia):
        print("Carregando banco vetorial existente...")
        vectorstore = Chroma(
            persist_directory=diretorio_persistencia,
            embedding_function=embeddings
        )
        return vectorstore

    print("Criando novo banco vetorial e gerando embeddings com HuggingFace...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    
    chunks = splitter.split_documents(documentos)
    print(f"Documentos fatiados em {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=diretorio_persistencia
    )
    
    return vectorstore