import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup


print("Coletando notícias via RSS do G1")

categorias = {
    "Economia": "https://g1.globo.com/rss/g1/economia/",
    "Política": "https://g1.globo.com/rss/g1/politica/",
    "Mundo": "https://g1.globo.com/rss/g1/mundo/",
    "Tecnologia": "https://g1.globo.com/rss/g1/tecnologia/",
    "Educação": "https://g1.globo.com/rss/g1/educacao/"
}

noticias = []

for categoria, feed_url in categorias.items():
    print(f"→ Coletando categoria: {categoria}")
    r = requests.get(feed_url)
    soup = BeautifulSoup(r.content, "xml")

    for item in soup.find_all("item"):
        titulo = item.title.text.strip() if item.title else ""
        link = item.link.text.strip() if item.link else ""
        descricao = item.description.text.strip() if item.description else ""
        noticias.append({
            "titulo": titulo,
            "descricao": descricao,
            "link": link,
            "categoria": categoria
        })

df = pd.DataFrame(noticias)
print(f"✅ {len(df)} notícias coletadas!\n")

print("----------------------------\n")
print("Limpando textos")

nltk.download("stopwords", quiet=True)
stopwords_pt = set(stopwords.words("portuguese"))

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"[^a-zA-Záéíóúâêîôûãõç\s]", "", texto)
    palavras = texto.split()
    palavras = [p for p in palavras if p not in stopwords_pt]
    return " ".join(palavras)

df["texto_limpo"] = df["titulo"].apply(limpar_texto)

print("----------------------------\n")
print("treinando modelo")

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["texto_limpo"])
y = df["categoria"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nResultados:")
print("Acurácia:", round(accuracy_score(y_test, y_pred), 3))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

df.to_csv("noticias_g1_rss.csv", index=False)
print("\n💾 Dados salvos em 'noticias_g1_rss.csv'")
