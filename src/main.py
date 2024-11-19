import streamlit as st
from pathlib import Path
from load_data import load_data, load_json_data, json_to_documents, create_documents_from_text
from rag_chain import (
    init_groq_client,
    split_documents_into_chunks,
    generate_embeddings,
    create_index_faiss,
    ml_block
)

def main():
    # Definir os caminhos
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "data" / "vestibular_unicamp_2025.json"
    model_name = 'all-MiniLM-L6-v2'

    # Título da aplicação Streamlit
    st.title("Chatbot RAG para Vestibular Unicamp 2025")

    # Inicializar o cliente do Groq
    client = init_groq_client()

    # Etapa 1: Carregar dados JSON
    text = load_data()
    json_data = load_json_data(json_path)

    # Etapa 2: Criar documentos a partir dos dados JSON
    docs = json_to_documents(json_data)
    
    # Etapa 3: Dividir documentos em partes menores
    splits = split_documents_into_chunks(docs)

    # Etapa 4: Gerar embeddings para as partes de texto
    chunk_texts = [chunk.page_content for chunk in splits]
    embeddings, model = generate_embeddings(model_name, chunk_texts)

    # Etapa 5: Criar índice FAISS
    index = create_index_faiss(embeddings)

    # Variável para armazenar o histórico de perguntas e respostas
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Etapa 6: Interface de entrada do usuário com Streamlit
    user_query = st.text_input("Digite sua pergunta:", "")
    
    if st.button("Enviar"):
        if user_query:
            # Rodar o pipeline completo para processar a pergunta
            st.write("Processando sua pergunta...")
            answer_rag = ml_block(user_query, model, index, chunk_texts, embeddings, client=client)

            # Armazenar a pergunta e resposta no histórico
            st.session_state.history.append({"pergunta": user_query, "resposta": answer_rag})

            # Exibir as respostas anteriores
            st.write("\n--- Respostas Anteriores ---")
            for i, item in enumerate(st.session_state.history):
                st.write(f"**Pergunta {i+1}:** {item['pergunta']}")
                st.write(f"**Resposta {i+1}:** {item['resposta']}")
        else:
            st.write("Por favor, insira uma pergunta.")

if __name__ == "__main__":
    main()
