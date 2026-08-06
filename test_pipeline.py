from src.loaders import carregar_documentos
from src.vectorstore import criar_ou_carregar_vectorstore
from src.rag_chain import criar_cadeia_rag

def main():
    print("--- 1. Carregando documentos da AutoFab ---")
    docs = carregar_documentos("data")
    print(f"Total de documentos/páginas processados: {len(docs)}")

    print("\n--- 2. Inicializando Banco Vetorial (ChromaDB) ---")
    vectorstore = criar_ou_carregar_vectorstore(docs)

    print("\n--- 3. Criando Cadeia RAG com Groq (Llama 3.1) ---")
    chain = criar_cadeia_rag(vectorstore)

    # Pergunta 1: Deve constar na base (Manutenção/CLP)
    pergunta_1 = "O que significa o código de erro ERR-101 no CLP?"
    print(f"\n[PERGUNTA 1]: {pergunta_1}")
    resposta_1 = chain.invoke(pergunta_1)
    print(f"[RESPOSTA AUTOFAB]:\n{resposta_1}")

    # Pergunta 2: Deve disparar a trava de segurança (Não existe na base)
    pergunta_2 = "Qual é a política de reembolso para viagens internacionais?"
    print(f"\n\n[PERGUNTA 2]: {pergunta_2}")
    resposta_2 = chain.invoke(pergunta_2)
    print(f"[RESPOSTA AUTOFAB]:\n{resposta_2}")

if __name__ == "__main__":
    main()