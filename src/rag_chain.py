import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROMPT_TEMPLATE_AUTOFAB = """
Você é o Agente de IA especialista da AutoFab (Manufatura Inteligente).
Sua missão é auxiliar os colaboradores respondendo dúvidas operacionais, normativas e técnicas de forma clara, direta e cortês.

DIRETRIZES RÍGIDAS DE RESPOSTA:
1. Responda à dúvida do colaborador UTILIZANDO EXCLUSIVAMENTE as informações fornecidas no CONTEXTO abaixo.
2. Não invente procedimentos, regras ou códigos que não estejam no contexto.
3. Se a resposta não estiver contida no contexto fornecido, responda exatamente:
   "Não encontrei essa informação na base de conhecimento oficial da AutoFab. Por favor, consulte seu supervisor ou o setor responsável."
4. Mantenha um tom profissional, didático e focado na indústria de manufatura.

CONTEXTO RECUPERADO:
{context}

PERGUNTA DO COLABORADOR:
{question}

RESPOSTA DO AGENTE AUTOFAB:
"""

def formatar_documentos(docs):
    """
    Junta o conteúdo dos chunks recuperados em um único texto para o prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def criar_cadeia_rag(vectorstore):
    """
    Instancia a LLM via Groq e monta a cadeia de QA usando LCEL (LangChain Expression Language).
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("A variável GROQ_API_KEY não foi configurada no arquivo .env!")

    # Instancia a LLM usando o modelo Llama 3.1 8B da Groq
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    # Configura o buscador no banco vetorial (traz os 3 chunks mais parecidos)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Cria o template do prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_AUTOFAB)

    # Monta a cadeia usando LCEL
    rag_chain = (
        {"context": retriever | formatar_documentos, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain