# api_rag.py (Versão FINAL e COMPLETA com DEBUGs para inicialização)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json

# --- Importações para FastAPI-MCP ---
from fastapi_mcp import FastApiMCP 

# --- Importações LangChain e ChromaDB ---
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama as Ollama_LLM

# --- Configurações RAG ---
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral" 

# --- Inicializa a aplicação FastAPI ---
app = FastAPI(
    title="API de Análise de Tickets RAG (Básico MCP)",
    description="API básica com MCP para testar integração com OpenWebUI."
)

# --- Configuração CORS ---
origins = [
    "http://localhost",
    "http://localhost:8080", 
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inicializa o FastApiMCP (forma mais básica) ---
mcp = FastApiMCP(app) 

# --- Variáveis globais para os componentes RAG ---
# Serão preenchidas durante o evento de startup
global_qa_chain = None
global_retriever = None
global_embeddings_model = None
global_vectorstore = None
global_llm_model_instance = None
global_query_cache = {} 

# --- Definir schema da requisição para o endpoint de consulta (mantido para testes diretos) ---
class QueryRequest(BaseModel):
    query: str

# --- REMOVIDO: CLASSE DE FERRAMENTAS MCP TicketAnalysisTools ---
# Esta classe foi removida para simplificar a inicialização e evitar erros.
# Vamos focar em expor o endpoint /query_rag diretamente.

# --- Função para inicializar os componentes RAG (chamada na inicialização da API) ---
@app.on_event("startup")
async def startup_event():
    """
    Função executada na inicialização da aplicação FastAPI.
    Aqui carregamos o ChromaDB, configuramos a cadeia de RAG.
    """
    global global_qa_chain, global_retriever, global_embeddings_model, global_vectorstore, global_llm_model_instance, global_query_cache
    
    print("\n[API Startup] Inicializando componentes RAG...")

    # DEBUG: Etapa 1.0 - Antes de inicializar modelo de embedding
    print("[API Startup] DEBUG: Entrando na inicialização do modelo de embedding...")
    # 1. Inicializar modelo de embedding
    print(f"[API Startup] Inicializando modelo de embedding '{EMBEDDING_MODEL}' com Ollama...")
    try:
        global_embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # DEBUG: Etapa 1.1 - Antes do teste de embedding
        print("[API Startup] DEBUG: Realizando teste de embedding...")
        global_embeddings_model.embed_query("startup test")
        print("[API Startup] DEBUG: Teste de embedding concluído com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO [API Startup]: Falha ao inicializar/testar o modelo de embedding Ollama: {e}")
        print("Certifique-se de que o Ollama Server está rodando e o modelo 'nomic-embed-text' foi baixado.")
        global_embeddings_model = None 

    # DEBUG: Etapa 2.0 - Antes de carregar ChromaDB
    print("[API Startup] DEBUG: Entrando no carregamento do ChromaDB...")
    # 2. Carregar ChromaDB
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"ERRO CRÍTICO [API Startup]: O diretório de persistência do ChromaDB '{PERSIST_DIRECTORY}' não foi encontrado.")
        print("Certifique-se de que a Fase 2 (ingestão de dados) foi concluída com sucesso.")
        global_vectorstore = None
    else:
        print(f"[API Startup] Carregando banco de dados ChromaDB de '{PERSIST_DIRECTORY}'...")
        try:
            global_vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=global_embeddings_model
            )
            print("[API Startup] DEBUG: ChromaDB instanciado.")
            # Pequeno teste de contagem para garantir que o Chroma responde
            print(f"[API Startup] ChromaDB carregado. Contém {global_vectorstore._collection.count()} documentos.")
            print("[API Startup] DEBUG: Contagem do ChromaDB verificada.")
        except Exception as e:
            print(f"ERRO CRÍTICO [API Startup]: Falha ao carregar o ChromaDB: {e}")
            print("Verifique a integridade da pasta 'chroma_db'.")
            global_vectorstore = None

    # DEBUG: Etapa 3.0 - Antes de configurar Retriever e LLM
    print("[API Startup] DEBUG: Entrando na configuração do Retriever e LLM...")
    # 3. Configurar Retriever e LLM para a cadeia de RAG
    if global_vectorstore and global_embeddings_model:
        global_retriever = global_vectorstore.as_retriever(search_kwargs={"k": 3})
        print("[API Startup] DEBUG: Retriever configurado.")
        
        print(f"[API Startup] Inicializando LLM '{LLM_MODEL}' com Ollama para Chain de QA...")
        try:
            global_llm_model_instance = Ollama_LLM(model=LLM_MODEL)
            print("[API Startup] DEBUG: LLM instanciado.")
        except Exception as e:
            print(f"ERRO CRÍTICO [API Startup]: Falha ao inicializar o LLM Ollama: {e}")
            print("Certifique-se de que o Ollama Server está rodando e o modelo '{LLM_MODEL}' foi baixado.")
            global_llm_model_instance = None

        if global_llm_model_instance:
            print("[API Startup] Configurando Chain de RAG...")
            try:
                global_qa_chain = RetrievalQA.from_chain_type(
                    llm=global_llm_model_instance,
                    chain_type="stuff",
                    retriever=global_retriever,
                    return_source_documents=True
                )
                print("[API Startup] Chain de RAG configurada com sucesso!")
            except Exception as e:
                print(f"ERRO CRÍTICO [API Startup]: Falha ao configurar a Chain de RAG: {e}")
                global_qa_chain = None
        else:
            global_qa_chain = None
    else:
        print("[API Startup] Retriever e/ou Vectorstore não puderam ser configurados devido a erros anteriores.")
        global_retriever = None
        global_qa_chain = None
    
    print("[API Startup] Componentes RAG inicializados.")

    # --- Montar o servidor MCP no final da inicialização ---
    mcp.mount() 
    print("[API Startup] Servidor MCP montado.")


# --- Endpoints da API ---
# Estes são endpoints FastAPI padrão. FastApiMCP pode tentar expô-los automaticamente.

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Análise de Tickets RAG! Use /docs para ver a documentação interativa."}

@app.get("/health")
async def health_check():
    status = "ok"
    messages = ["API está funcionando corretamente."]
    
    if not global_embeddings_model:
        status = "degraded"
        messages.append("Erro: Modelo de Embedding (Ollama) não inicializado.")
    if not global_vectorstore:
        status = "degraded"
        messages.append("Erro: ChromaDB não carregado ou persist_directory ausente.")
    if not global_llm_model_instance:
        status = "degraded"
        messages.append("Erro: LLM (Ollama) não inicializado.")
    if not global_qa_chain:
        status = "degraded"
        messages.append("Erro: Chain de RAG não configurada.")

    return {"status": status, "messages": messages}

@app.post("/query_rag") # Este endpoint pode ser o que o MCP tentará expor
async def query_rag(request: QueryRequest):
    """
    Endpoint para consultar a base de conhecimento RAG.
    Recebe uma pergunta e retorna a resposta do LLM com base nos documentos relevantes.
    """
    global global_qa_chain, global_query_cache 
    
    if not global_qa_chain:
        raise HTTPException(status_code=503, detail="Serviço RAG não está totalmente inicializado. Verifique os logs de startup.")

    query = request.query
    print(f"\n[API] Recebida query: '{query}'")

    if query in global_query_cache:
        print(f"[API] Resultado recuperado do cache para a query: '{query}'")
        cached_response = global_query_cache[query]
        return cached_response 

    try:
        response = global_qa_chain.invoke({"query": query})
        
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
        
        global_query_cache[query] = result_to_cache
        
        print(f"[API] Query '{query}' processada com sucesso.")
        return result_to_cache

    except Exception as e:
        print(f"ERRO [API]: Falha ao processar a query RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a query RAG: {e}")

# --- REMOVIDO: Endpoint de Manifesto do Plugin (/well-known/ai-plugin.json) ---
# Este endpoint é desnecessário, pois o mcpo gerará a documentação OpenAPI diretamente.

#if __name__ == "__main__":
    import uvicorn
    # A API FastApi agora rodará na porta 8001 para não conflitar com o mcpo na 8000
    uvicorn.run("api_rag:app", host="0.0.0.0", port=8001)