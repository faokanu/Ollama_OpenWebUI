# ingestao_dados.py

import pandas as pd
import os
import uuid

# --- Importações LangChain e ChromaDB ---
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document # Importar Document explicitly

# --- Função de Carregamento e Formatação (EXISTENTE) ---
def carregar_e_formatar_documentos(diretorio_csv: str, colunas_conteudo: list) -> list:
    """
    Carrega arquivos CSV de um diretório, gera um ID único para cada linha
    e formata o conteúdo de cada linha em um documento de texto.

    Args:
        diretorio_csv (str): Caminho para o diretório contendo os arquivos CSV.
        colunas_conteudo (list): Lista de nomes das colunas cujo conteúdo deve ser
                                 incluído no texto do documento.

    Returns:
        list: Uma lista de dicionários, onde cada dicionário representa um documento
              formatado com seu ID e conteúdo. Ex:
              [{'id': '...', 'content': '...'}, {'id': '...', 'content': '...'}]
    """
    documentos_formatados = []
    id_contador = 0

    print(f"Lendo arquivos CSV de: {diretorio_csv}")
    for nome_arquivo in os.listdir(diretorio_csv):
        if nome_arquivo.endswith('.csv'):
            caminho_completo = os.path.join(diretorio_csv, nome_arquivo)
            print(f"Processando arquivo: {nome_arquivo}")
            try:
                try:
                    df = pd.read_csv(caminho_completo, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(caminho_completo, encoding='latin1')
                except Exception as e:
                    print(f"Erro de encoding ao ler {nome_arquivo}: {e}. Tentando sem especificar encoding.")
                    df = pd.read_csv(caminho_completo)

                colunas_validas = [col for col in colunas_conteudo if col in df.columns]
                if not colunas_validas:
                    print(f"Aviso: Nenhuma coluna de conteúdo válida encontrada em {nome_arquivo}. Pulando.")
                    continue

                for index, row in df.iterrows():
                    id_contador += 1
                    document_id = f"doc_{id_contador}" # Ou str(uuid.uuid4())

                    texto_documento = f"## Documento ID: {document_id}\n"
                    
                    for col_name in colunas_validas:
                        value = row.get(col_name)
                        if pd.notna(value):
                            texto_documento += f"**{col_name.replace('_', ' ').title()}**: {value}\n"
                    
                    texto_documento += "\n"

                    documentos_formatados.append(
                        {'id': document_id, 'content': texto_documento, 'source': nome_arquivo}
                    )

            except Exception as e:
                print(f"Erro ao ler ou processar {nome_arquivo}: {e}")

    return documentos_formatados

# --- FUNÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # --- CONFIGURAÇÃO ---
    DIRETORIO_DOS_CSVS = 'data' # Ajuste para o nome da sua pasta de dados
    NOME_DO_ARQUIVO_CSV = 'tickets.csv' # Nome específico do arquivo dentro da pasta

    COLUNAS_PARA_CONTEUDO = ['subject', 'body', 'answer', 'type', 'queue', 'priority', 'language', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
    
    # --- CONFIGURAÇÕES DE CHUNKING E EMBEDDING ---
    CHUNK_SIZE = 1000  # Tamanho máximo de caracteres por chunk
    CHUNK_OVERLAP = 200 # Quantidade de caracteres que se sobrepõem entre chunks
    EMBEDDING_MODEL = "nomic-embed-text" # Modelo de embedding a ser usado com Ollama
    
    # --- CONFIGURAÇÃO DO CHROMADB ---
    # Caminho onde o ChromaDB irá armazenar os dados. Será uma pasta dentro do seu projeto.
    PERSIST_DIRECTORY = "./chroma_db" 

    # --- EXECUÇÃO ---
    if not os.path.exists(DIRETORIO_DOS_CSVS):
        print(f"Erro: O diretório '{DIRETORIO_DOS_CSVS}' não foi encontrado.")
        print("Por favor, crie esta pasta e coloque seus arquivos CSV dentro dela (ex: 'tickets.csv').")
    else:
        # Carrega e formata os documentos brutos
        documentos_processados_dicts = carregar_e_formatar_documentos(
            DIRETORIO_DOS_CSVS,
            COLUNAS_PARA_CONTEUDO
        )

        if not documentos_processados_dicts:
            print("Nenhum documento foi processado. Verifique os arquivos CSV e as configurações.")
        else:
            print(f"\nTotal de {len(documentos_processados_dicts)} documentos brutos processados.")
            
            ### NOVIDADE: REDUZINDO DOCUMENTOS PARA TESTE (DESCOMENTE PARA TESTAR RAPIDAMENTE) ###
            documentos_processados_dicts = documentos_processados_dicts[:100] 
            # print(f"DEBUG: Limitando a {len(documentos_processados_dicts)} documentos para teste rápido.")

            langchain_documents = []
            for doc_dict in documentos_processados_dicts:
                langchain_documents.append(
                    Document(page_content=doc_dict['content'], metadata={'id': doc_dict['id'], 'source': doc_dict['source']})
                )
            print(f"Convertidos {len(langchain_documents)} documentos brutos para formato LangChain Document.")


            # --- 2. Fragmentar (Chunking) os Documentos ---
            print(f"Dividindo documentos em chunks (tamanho: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True 
            )
            chunks = text_splitter.split_documents(langchain_documents)
            print(f"Total de chunks gerados: {len(chunks)}")
            print("\nExemplo do primeiro chunk gerado:\n")
            print("--- INÍCIO DO EXEMPLO DO CHUNK ---")
            print(chunks[0].page_content)
            print(chunks[0].metadata)
            print("--- FIM DO EXEMPLO DO CHUNK ---")

            # --- 3. Gerar Embeddings com Ollama ---
            print(f"\nInicializando modelo de embedding '{EMBEDDING_MODEL}' com Ollama...")
            try:
                embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
                ### NOVIDADE: TESTE DE EMBEDDING ANTES DA INGESTÃO EM MASSA ###
                print("Realizando um teste rápido de embedding...")
                test_text = "Isso é um texto de teste para o embedding."
                embedding_test = embeddings_model.embed_query(test_text)
                print(f"Teste de embedding bem-sucedido. Tamanho do vetor: {len(embedding_test)} dimensões.")
                print("Continuando com a ingestão em massa...")
            except Exception as e:
                print(f"ERRO CRÍTICO: Falha ao inicializar/testar o modelo de embedding Ollama: {e}")
                print("A ingestão não pode continuar. Verifique:")
                print("1. Se o Ollama Server está rodando no seu terminal (ex: 'ollama serve').")
                print("2. Se o modelo 'nomic-embed-text' foi baixado corretamente ('ollama pull nomic-embed-text').")
                print("3. Se o Ollama tem recursos de memória suficientes.")
                exit() 

            # --- 4. Armazenar no ChromaDB ---
            print(f"\nArmazenando {len(chunks)} chunks e embeddings no ChromaDB em '{PERSIST_DIRECTORY}'...")
            try:
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings_model,
                    persist_directory=PERSIST_DIRECTORY
                )
                vectorstore.persist() 
                print("\nDados armazenados com sucesso no ChromaDB.")
                print(f"Número de documentos no ChromaDB: {vectorstore._collection.count()}")
                print(f"ChromaDB persistido em: {os.path.abspath(PERSIST_DIRECTORY)}")

            except Exception as e:
                print(f"ERRO CRÍTICO: Falha ao armazenar dados no ChromaDB: {e}")
                print("Verifique se há permissões de escrita no diretório e se o ChromaDB está configurado corretamente.")
            
            print("\nFASE 2 (Injestão de Dados) CONCLUÍDA COM SUCESSO!")