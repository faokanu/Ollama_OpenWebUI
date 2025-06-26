import ollama

try:
    response = ollama.chat(model='mistral', messages=[
        {"role": "user", "content": "Qual a capital do Parana?"}
    ])
    print(response['message']['content'])
except Exception as e:
    print(f"Ocorreu um erro: {e}")
    print("Verifique se o servidor Ollama está rodando e se o modelo 'Mistral' foi baixado corretamente.")
    print("Você pode baixar o modelo com o comando: `ollama pull mistral`")