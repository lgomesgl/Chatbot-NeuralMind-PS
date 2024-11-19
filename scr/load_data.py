import requests
import urllib3
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from langchain.schema import Document

import requests
from bs4 import BeautifulSoup
import re
import json
import urllib3

# Função para desativar alertas de SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_data():
    # URL da página
    base_dir = Path(__file__).resolve().parent.parent
    url = "https://www.pg.unicamp.br/norma/31879/0"
    
    # Fazer requisição GET
    response = requests.get(url, verify=False)
    
    # Parse do conteúdo HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Estrutura inicial do JSON
    document = {"Chapters": []}
    current_chapter = None
    current_article = None

    def parse_incidences(paragraph):
        """Função para identificar e organizar os incisos dentro de um parágrafo"""
        incidence_pattern = re.compile(r'^\s*(§\d+)\s*(.*)')
        incidences = []
        lines = paragraph.split('\n')

        for line in lines:
            match = incidence_pattern.match(line.strip())
            if match:
                incidences.append({
                    "Inciso": match.group(1),
                    "Text": match.group(2)
                })
        
        return incidences if incidences else None

    # Procurar os elementos relevantes no HTML
    elements = soup.find_all(['p', 'li'])

    # Organizar os dados conforme necessário
    chapter_data = None
    article_data = None

    text_ = ''
    for elem in elements:
        text = elem.get_text().strip()
        text_ += text + '\n'  # Adiciona uma quebra de linha após cada elemento
        # Identificar capítulos e artigos
        if text.startswith("Capítulo"):
            if chapter_data:
                document["Chapters"].append(chapter_data)
            chapter_data = {
                "Chapter": text,
                "Articles": []
            }
        elif text.startswith("Art."):
            if article_data:
                chapter_data["Articles"].append(article_data)
            article_data = {
                "Article": text,
                "Text": "",
                "Incidences": [],
                "Paragraphs": []
            }
        elif text.startswith("§"):
            # Organizar incisos e parágrafos
            incidences = parse_incidences(text)
            if incidences:
                article_data["Incidences"].extend(incidences)
            else:
                article_data["Paragraphs"].append({
                    "Text": text
                })
        else:
            # Adicionando o texto ao artigo
            if article_data:
                article_data["Text"] += text

    # Adicionando o último artigo e capítulo
    if article_data:
        chapter_data["Articles"].append(article_data)
    if chapter_data:
        document["Chapters"].append(chapter_data)

    # Salvar como arquivo JSON
    with open(base_dir / "data" / "vestibular_unicamp_2025.json", "w", encoding="utf-8") as json_file:
        json.dump(document, json_file, ensure_ascii=False, indent=4)

    return text_

def load_json_data(json_path: str):
    """
    Loads JSON data from the specified file path.

    Parameters:
    - json_path (str): Path to the JSON file.

    Returns:
    - dict: Parsed JSON data.
    """
    with open(json_path, "r", encoding='utf-8') as file:
        return json.load(file)

def json_to_documents(data):
    documents = []
    
    # Iterar sobre os capítulos, artigos e incisos
    for chapter in data["Chapters"]:
        chapter_text = chapter["Chapter"]
        
        # Adicionar um documento para o capítulo
        documents.append(Document(
            page_content=chapter_text,
            metadata={"type": "chapter", "title": chapter_text}
        ))
        
        for article in chapter["Articles"]:
            article_text = article["Text"]
            
            # Adicionar o texto do artigo como conteúdo do documento
            documents.append(Document(
                page_content=article_text,  # Texto do artigo no campo page_content
                metadata={"type": "article", "title": article["Article"]}
            ))
            
            # Adicionar os incisos como documentos individuais
            for incidence in article["Incidences"]:
                incidence_text = incidence["Text"]
                documents.append(Document(
                    page_content=incidence_text,
                    metadata={"type": "inciso", "inciso": incidence["Inciso"], "article": article["Article"]}
                ))
                
            # Adicionar os parágrafos como documentos individuais
            for paragraph in article["Paragraphs"]:
                paragraph_text = paragraph["Text"]
                documents.append(Document(
                    page_content=paragraph_text,
                    metadata={"type": "paragraph", "article": article["Article"]}
                ))
    
    return documents

def create_documents_from_text(text):
    """
    Função que cria documentos a partir de um texto unificado.
    A função busca separar o texto em capítulos, artigos, incisos e parágrafos e
    cria um Document para cada uma dessas partes.
    
    Parâmetros:
    - text: O texto unificado que contém os dados estruturados.

    Retorna:
    - Uma lista de objetos Document.
    """
    documents = []
    
    # Dividir o texto em partes com base em um padrão de estrutura (pode ser ajustado conforme necessário)
    chapters = text.split('Capítulo:')  # Divida o texto por "Capítulo:" ou outros delimitadores
    
    # Iterar sobre as seções (capítulos)
    for chapter in chapters:
        if chapter.strip():  # Ignorar se o texto estiver vazio
            # Criar um Document para cada capítulo
            chapter_title = 'Capítulo: ' + chapter.split('\n')[0].strip()  # Pega o título do capítulo
            chapter_text = chapter.strip()
            documents.append(Document(
                page_content=chapter_text,
                metadata={"type": "chapter", "title": chapter_title}
            ))

            # Dividir o capítulo em artigos (presumindo que cada artigo começa com "Artigo:")
            articles = chapter.split('Artigo:')
            for article in articles:
                if article.strip():
                    # Criar um Document para cada artigo
                    article_title = 'Artigo: ' + article.split('\n')[0].strip()  # Pega o título do artigo
                    article_text = article.strip()
                    documents.append(Document(
                        page_content=article_text,
                        metadata={"type": "article", "title": article_title}
                    ))

                    # Dividir o artigo em incisos (presumindo que cada inciso começa com "Inciso:")
                    incisos = article.split('Inciso:')
                    for inciso in incisos:
                        if inciso.strip():
                            # Criar um Document para cada inciso
                            inciso_text = inciso.strip()
                            documents.append(Document(
                                page_content=inciso_text,
                                metadata={"type": "inciso", "text": inciso_text}
                            ))

    return documents

# load_data()
# json_data = load_json_data(r'D:\LUCAS\PROJECTS\Chatbot-NeuralMind\data\vestibular_unicamp_2025.json')
# # docs = process_json_to_documents(json_data=json_data)
# docs = json_to_documents(json_data=json_data)


