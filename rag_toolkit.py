"""
title: Análise de Tickets RAG
author: Seu Nome
author_url: https://seusite.com (Opcional)
description: Ferramenta para consultar uma base de conhecimento de protocolos de atendimento de TI e obter informações detalhadas sobre tickets, problemas, soluções, status e histórico de atendimentos.
required_open_webui_version: 0.4.0 # Verifique a versão do seu OpenWebUI
version: 1.0.0
license: MIT
requirements: 
    langchain==0.2.5, # Versões específicas para garantir compatibilidade
    langchain-community==0.2.5,
    chromadb==0.4.24,
    pandas==2.2.2,
    # langchain-ollama==0.1.0 # Se você migrar para as novas classes Ollama (opcional por enquanto)
"""

import os
import json
from datetime import datetime
from pydantic import BaseModel, Field

# --- Importações LangChain e ChromaDB ---
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama as Ollama_LLM

# --- Configurações RAG (As mesmas usadas em ingestao_dados.py) ---
PERSIST_DIRECTORY = "./chroma_db" # Caminho relativo à raiz do seu projeto/onde você roda Docker
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral" 

# --- Classe Tools: Contém todas as ferramentas e lógica de inicialização ---
class Tools:
    def __init__(self):
        """
        Inicializa a ferramenta e carrega os componentes RAG (ChromaDB, modelos Ollama).
        Este método é chamado uma vez quando o OpenWebUI carrega o toolkit.
        """
        self.qa_chain = None
        self.query_cache = {} # Cache para a instância desta ferramenta

        print("\n[RAG Toolkit Startup] Iniciando componentes RAG para o Toolkit...")

        try:
            # 1. Inicializar modelo de embedding
            # Acessando Ollama do Docker: host.docker.internal geralmente funciona
            # Ou o IP da sua máquina se o modo de rede do Docker for diferente
            self.embeddings_model = OllamaEmbeddings(
                model=EMBEDDING_MODEL, 
                base_url="http://host.docker.internal:11434" # APONTE PARA SEU OLLAMA
            ) 
            self.embeddings_model.embed_query("toolkit startup test")
            print("[RAG Toolkit Startup] Modelo de embedding Ollama inicializado e testado.")

            # 2. Carregar ChromaDB
            # O caminho 'PERSIST_DIRECTORY' deve ser acessível de DENTRO do contêiner OpenWebUI.
            # Se 'chroma_db' está no seu host, você DEVE mapear este volume no Docker.
            if not os.path.exists(PERSIST_DIRECTORY):
                print(f"ERRO [RAG Toolkit Startup]: Diretório ChromaDB '{PERSIST_DIRECTORY}' não encontrado. Verifique o mapeamento de volume Docker.")
                return # Não inicializa a cadeia QA
            
            self.vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings_model 
            )
            print(f"[RAG Toolkit Startup] ChromaDB carregado. Contém {self.vectorstore._collection.count()} documentos.")

            # 3. Inicializar LLM para a cadeia de RAG
            self.llm_model_instance = Ollama_LLM(
                model=LLM_MODEL, 
                base_url="http://host.docker.internal:11434" # APONTE PARA SEU OLLAMA
            ) 
            
            # 4. Configurar Retriever e Cadeia de RAG
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_model_instance,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            print("[RAG Toolkit Startup] Cadeia de RAG configurada com sucesso!")

        except Exception as e:
            print(f"ERRO CRÍTICO [RAG Toolkit Startup]: Falha ao inicializar o Toolkit RAG: {e}")
            self.qa_chain = None # Garante que a ferramenta não será usada se a inicialização falhar

    def consulta_base_conhecimento( # Não há decorador @Field() nesta linha
        self, 
        query: str = Field(..., description="A pergunta do usuário sobre os tickets.") # O description do Field é para o parâmetro
    ) -> str:
        """
        Consulta a base de conhecimento de tickets de TI para obter informações relevantes
        e gera uma resposta baseada nessas informações.

        Args:
            query (str): A pergunta do usuário sobre os tickets.

        Returns:
            str: A resposta gerada pelo LLM, incluindo as fontes se aplicável, formatada como JSON string.
        """
        if not self.qa_chain:
            return json.dumps({"result": "Serviço de base de conhecimento não inicializado. Verifique os logs do Toolkit.", "source_documents": []})

        print(f"\n[RAG Tool] Recebida query: '{query}'")

        if query in self.query_cache:
            print(f"[RAG Tool] Resultado recuperado do cache para a query: '{query}'")
            cached_response = self.query_cache[query]
            return json.dumps(cached_response)

        try:
            response = self.qa_chain.invoke({"query": query})
            
            formatted_sources = []
            if response.get('source_documents'):
                for doc in response['source_documents']:
                    formatted_sources.append({
                        "id": doc.metadata.get('id', 'N/A'),
                        "source": doc.metadata.get('source', 'N/A'),
                        "content_snippet": doc.page_content[:200] + "..."
                    })
            
            result_to_cache = {
                "query": query,
                "result": response.get('result', "Nenhuma resposta gerada."),
                "source_documents": formatted_sources
            }
            
            self.query_cache[query] = result_to_cache
            
            print(f"[RAG Tool] Query '{query}' processada com sucesso.")
            return json.dumps(result_to_cache)

        except Exception as e:
            print(f"ERRO [RAG Tool]: Falha ao processar a query RAG: {e}")
            return json.dumps({"result": f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}", "source_documents": []})