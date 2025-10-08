# EconomIA
Trabalho de IA

Este projeto coleta notícias do site G1 via RSS, realiza pré-processamento de texto, gera embeddings usando o modelo Sentence-BERT e treina um classificador KNN para categorizar as notícias.

🔹 Funcionalidades

1. Coleta notícias de várias categorias do G1:

  - **Economia**
  - **Política**
  - **Mundo**
  - **Tecnologia**
  - **Educação**

2. Limpa os textos (remove URLs, pontuação e stopwords em português).

3. Gera embeddings com SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2).

4. Treina um modelo KNN usando os embeddings.

5. Avalia o modelo com acurácia, precisão, recall e F1-score.

6. Salva os dados coletados em um arquivo CSV (noticias_g1_rss.csv).

🔹 Requisitos / Dependências

Certifique-se de ter Python >= 3.8 e instalar as bibliotecas abaixo:
```bash
pip install requests pandas beautifulsoup4 nltk scikit-learn sentence-transformers
```

O NLTK precisa baixar a lista de stopwords em português:
```bash
import nltk
nltk.download("stopwords")
```
🔹 Como rodar

Clone ou baixe este repositório.

Abra o terminal na pasta onde está o arquivo pipeline.py.

Execute o script:
```bash
python pipeline.py
```

O que o script faz:

- Coleta notícias de cada categoria via RSS.
- Limpa os textos.
- Gera embeddings usando Sentence-BERT.
- Treina o modelo KNN.
- Imprimi os resultados de acurácia e relatório de classificação.
- Salva os dados em noticias_g1_rss.csv.
