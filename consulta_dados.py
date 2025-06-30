# consulta_dados.py

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA # Para uma cadeia de QA mais completa no futuro
from langchain_community.llms import Ollama as Ollama_LLM # Renomeado para evitar conflito com embeddings

# --- CONFIGURAÇÃO ---
# O mesmo diretório de persistência usado na ingestão
PERSIST_DIRECTORY = "./chroma_db" 
EMBEDDING_MODEL = "nomic-embed-text" # O mesmo modelo de embedding usado na ingestão
LLM_MODEL = "mistral" # O modelo LLM que você está rodando no Ollama

# Cache de consultas (para esta demonstração, um dicionário simples em memória)
query_cache = {}

def get_rag_chain():
    """
    Inicializa o modelo de embedding, carrega o ChromaDB e configura o retriever.
    """
    print(f"Inicializando modelo de embedding '{EMBEDDING_MODEL}' para consulta...")
    try:
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"ERRO: Falha ao inicializar o modelo de embedding Ollama: {e}")
        print("Certifique-se de que o Ollama Server está rodando e o modelo 'nomic-embed-text' foi baixado.")
        return None, None

    print(f"Carregando banco de dados ChromaDB de '{PERSIST_DIRECTORY}'...")
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"ERRO: O diretório de persistência do ChromaDB '{PERSIST_DIRECTORY}' não foi encontrado.")
        print("Certifique-se de que a Fase 2 (ingestão de dados) foi concluída com sucesso.")
        return None, None

    try:
        # Carrega o banco de dados ChromaDB existente
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings_model # É crucial usar a mesma função de embedding!
        )
        print(f"ChromaDB carregado. Contém {vectorstore._collection.count()} documentos.")
    except Exception as e:
        print(f"ERRO: Falha ao carregar o ChromaDB: {e}")
        print("Verifique a integridade da pasta 'chroma_db'.")
        return None, None
    
    # Configura o retriever (busca por similaridade)
    # search_kwargs controla quantos chunks serão retornados (k=N)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    
    print(f"Inicializando LLM '{LLM_MODEL}' com Ollama para Chain de QA...")
    try:
        llm = Ollama_LLM(model=LLM_MODEL)
    except Exception as e:
        print(f"ERRO: Falha ao inicializar o LLM Ollama: {e}")
        print("Certifique-se de que o Ollama Server está rodando e o modelo '{LLM_MODEL}' foi baixado.")
        return None, None

    # Configura a cadeia de RAG (Retrieval-Augmented Generation)
    # Esta cadeia pega a pergunta, busca chunks relevantes e usa o LLM para gerar uma resposta.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" significa que ele coloca todos os chunks no prompt do LLM
        retriever=retriever,
        return_source_documents=True # Para ver de onde a informação veio
    )

    print("Chain de RAG configurada com sucesso!")
    return qa_chain, retriever

def query_knowledge_base(qa_chain: RetrievalQA, retriever, query: str) -> dict:
    """
    Consulta a base de conhecimento e retorna a resposta e as fontes.
    Implementa um cache simples em memória.

    Args:
        qa_chain (RetrievalQA): A cadeia de RAG configurada.
        retriever: O retriever para buscar documentos.
        query (str): A pergunta do usuário.

    Returns:
        dict: Um dicionário contendo a resposta ('result') e os documentos fonte ('source_documents').
    """
    # Verifica o cache primeiro
    if query in query_cache:
        print(f"Resultado recuperado do cache para a query: '{query}'")
        return query_cache[query]

    print(f"\nProcessando nova query: '{query}'")
    
    # Primeiro, buscar os documentos relevantes (para mostrar ao usuário e entender o RAG)
    print("Buscando documentos relevantes na base de conhecimento...")
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        print("Nenhum documento relevante encontrado para a consulta. O LLM responderá com conhecimento geral.")
    else:
        print(f"Encontrados {len(relevant_docs)} documentos relevantes:")
        for i, doc in enumerate(relevant_docs):
            print(f"--- Documento {i+1} (ID: {doc.metadata.get('id', 'N/A')}, Fonte: {doc.metadata.get('source', 'N/A')}) ---")
            print(doc.page_content[:200] + "...") # Mostra os primeiros 200 caracteres
            print("-" * 50)


    # Agora, use a cadeia de RAG para obter a resposta do LLM
    print("Gerando resposta com o LLM e os documentos relevantes...")
    try:
        # A cadeia QA fará a busca e a geração da resposta combinada
        response = qa_chain.invoke({"query": query})
        
        # Armazena no cache antes de retornar
        query_cache[query] = response
        return response
    except Exception as e:
        print(f"ERRO ao gerar resposta com LLM: {e}")
        print("Verifique se o LLM está carregado corretamente no Ollama e se há recursos suficientes.")
        return {"result": "Desculpe, não foi possível gerar uma resposta no momento.", "source_documents": []}


if __name__ == "__main__":
    qa_chain, retriever = get_rag_chain()

    if qa_chain and retriever:
        print("\n--- Sistema de Consulta RAG Pronto ---")
        print("Digite sua pergunta ou 'sair' para encerrar.")
        
        while True:
            user_query = input("\nSua pergunta: ")
            if user_query.lower() == 'sair':
                print("Encerrando o sistema de consulta.")
                break
            
            response = query_knowledge_base(qa_chain, retriever, user_query)
            
            print("\n--- RESPOSTA ---")
            print(response['result'])
            
            if response.get('source_documents'):
                print("\n--- FONTES CONSULTADAS ---")
                for i, doc in enumerate(response['source_documents']):
                    print(f"[{i+1}] ID: {doc.metadata.get('id', 'N/A')}, Fonte: {doc.metadata.get('source', 'N/A')}")
                    # print(f"Conteúdo: {doc.page_content[:150]}...") # Opcional: mostrar um snippet da fonte
            print("------------------")
    else:
        print("\nFalha ao inicializar o sistema RAG. Por favor, corrija os erros acima.")