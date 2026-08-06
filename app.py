import streamlit as st
from src.loaders import carregar_documentos
from src.vectorstore import criar_ou_carregar_vectorstore
from src.rag_chain import criar_cadeia_rag

# Configuração da página
st.set_page_config(
    page_title="AutoFab - Assistente de Manufatura Inteligente",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 AutoFab — Agente de IA Corporativo")
st.subheader("Base de Conhecimento Centralizada da Manufatura")

# Cache do pipeline para evitar reprocessamento a cada mensagem
@st.cache_resource
def inicializar_agente():
    documentos = carregar_documentos("data")
    vectorstore = criar_ou_carregar_vectorstore(documentos)
    cadeia_rag = criar_cadeia_rag(vectorstore)
    return cadeia_rag

try:
    agente = inicializar_agente()
except Exception as e:
    st.error(f"Erro ao inicializar o agente de IA: {e}")
    st.stop()

# Gerenciamento de histórico no estado da sessão (session_state)
if "historico_mensagens" not in st.session_state:
    st.session_state.historico_mensagens = []

# Exibe mensagens anteriores
for msg in st.session_state.historico_mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada de pergunta do usuário
if prompt := st.chat_input("Faça uma pergunta sobre manutenção, qualidade ou normas operacionais..."):
    # Adiciona a pergunta do usuário ao histórico e exibe na tela
    st.session_state.historico_mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta via modelo Llama 3.1
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento oficial da AutoFab..."):
            resposta = agente.invoke(prompt)
            st.markdown(resposta)
    
    # Salva a resposta do assistente no histórico
    st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})