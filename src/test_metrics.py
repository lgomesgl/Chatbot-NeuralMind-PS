from load_data import load_json_data, json_to_documents
from rag_chain import (
    init_groq_client,
    split_documents_into_chunks,
    generate_embeddings,
    create_index_faiss,
    ml_block
)
from metrics.generation import rouge_score, bleu_score
from pathlib import Path

def main(question: str):
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "data" / "vestibular_unicamp_2025.json"
    model_name = 'all-MiniLM-L6-v2'
    
    client = init_groq_client()
    
    json_data = load_json_data(json_path)

    docs = json_to_documents(json_data)

    splits = split_documents_into_chunks(docs)

    chunk_texts = [chunk.page_content for chunk in splits]
    embeddings, model = generate_embeddings(model_name, chunk_texts)

    index = create_index_faiss(embeddings)
    
    answer_rag = ml_block(user_query=question, model=model, index=index, chunk_texts=chunk_texts, embeddings=embeddings, client=client)

    return answer_rag

questions = {
    "Quantas vagas regulares no total são oferecidas no vestibular 2025?": "3340 vagas",
    "Quantas vagas regulares no total são oferecidas pelo Edital ENEM-Unicamp 2025?": "314 vagas",
    "Como as vagas são distribuídas no vestibular?": "Vestibular Unicamp (VU) 2025, Edital ENEM-Unicamp 2025, Provão Paulista 2025,Vestibular Indígena,Edital de olimpíadas científicas e competições de conhecimento de áreas específica",
    "Quantas vagas são oferecidas pelo Edital de olimíadas científicas": "115 vagas",
    "Como se inscrever?": "Acessar o site da Unicamp (www.unicamp.br), Anexar os documentos necessários, Diploma ou Certificado de Conclusão do ensino médio ou equivalente, Efetuar o pagamento da taxa de inscrição.",
    "Existe algum tipo de programa de inclusão?": "Programa de Inclusão de Estudantes com Deficiência (PIED), Programa de Ação Afirmativa e Inclusão Social (PAAIS)"
}

# Evaluate generated responses
for question, reference_response in questions.items():
    print(f"Question: {question}")
    generated_response = main(question)

    # Calculate ROUGE and BLEU scores
    rouge_scores = rouge_score(generated_response, reference_response)
    bleu = bleu_score(generated_response, reference_response)

    # Display results
    print(f"Generated Response: {generated_response}")
    print(f"ROUGE Scores: {rouge_scores}")
    print(f"BLEU Score: {bleu}")
    print("-" * 50)

