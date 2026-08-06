import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)

def carregar_documentos(diretorio_data: str):
    """
    Varre a pasta de dados e carrega todos os arquivos utilizando 
    o carregador adequado para cada extensão.
    """
    documentos = []

    if not os.path.exists(diretorio_data):
        print(f"Diretório {diretorio_data} não encontrado.")
        return documentos

    for arquivo in os.listdir(diretorio_data):
        caminho_completo = os.path.join(diretorio_data, arquivo)
        
        # Ignora pastas ou arquivos ocultos/temporários
        if os.path.isdir(caminho_completo) or arquivo.startswith('.') or arquivo.startswith('~$'):
            continue

        extensao = os.path.splitext(arquivo)[1].lower()

        try:
            if extensao == '.pdf':
                loader = PyPDFLoader(caminho_completo)
                documentos.extend(loader.load())
            elif extensao == '.md':
                loader = UnstructuredMarkdownLoader(caminho_completo)
                documentos.extend(loader.load())
            elif extensao == '.csv':
                loader = CSVLoader(caminho_completo, encoding='utf-8')
                documentos.extend(loader.load())
            elif extensao in ['.json', '.txt']:
                # Carregamento simplificado e direto para JSON/TXT
                loader = TextLoader(caminho_completo, encoding='utf-8')
                documentos.extend(loader.load())
            elif extensao in ['.html', '.htm']:
                loader = BSHTMLLoader(caminho_completo)
                documentos.extend(loader.load())
            elif extensao == '.docx':
                loader = UnstructuredWordDocumentLoader(caminho_completo)
                documentos.extend(loader.load())
            elif extensao == '.pptx':
                loader = UnstructuredPowerPointLoader(caminho_completo)
                documentos.extend(loader.load())
            else:
                print(f"Formato não suportado ignorado: {arquivo}")

        except Exception as e:
            print(f"Erro ao carregar o arquivo {arquivo}: {e}")

    return documentos