import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')

"""## P3- Matriz de similitudes
> Elabore una matriz de similitud de coseno entre los documentos de la colección "El Señor de los Anillos". Debe aplicar los pesos TF-IDF.

### 1- Preprocesamiento
"""


def preprocesamiento(texto):
    # tokenizar
    tokens = nltk.word_tokenize(texto.lower())

    # filtrar stopwords
    with(open('keys/stoplist.txt')) as file:
        stop_list = [line.lower().strip() for line in file]
    stop_list += ['.', '?', '¿', '-', '!', '\'', ',', ':', '«', '(', ')',
                  '``', 'con', ';']

    # reducir palabras
    tokens_limpios = []
    for token in tokens:
        if token.lower() not in stop_list:
            tokens_limpios.append(token)

    # extraer raiz y contar frecuencia
    stemmer = SnowballStemmer('spanish')
    termino_tf = {}
    for w in tokens_limpios:
        stemmed = stemmer.stem(w)
        if stemmed in termino_tf:
            termino_tf[stemmed] += 1
        else:
            termino_tf[stemmed] = 1
    return termino_tf


textos = ["libro1.txt", "libro2.txt", "libro3.txt", "libro4.txt", "libro5.txt", "libro6.txt"]
textos_procesados = []
indice = {}
for file_name in textos:
    file = open("docs/" + file_name)
    texto = file.read().rstrip()
    texto = preprocesamiento(texto)
    textos_procesados.append(texto)

"""### 2- Similitud de coseno"""


def compute_tfidf(collection):
    # calcular los pesos TF_IDF para cada documento de la coleccion
    stemmer = SnowballStemmer('spanish')
    # df: en cuantos documentos aparece cada término
    termino_df = {}
    terminos = set()
    for doc in collection:
        for termino in doc:
            terminos.add(termino)
            if termino in termino_df:
                termino_df[termino] += 1
            else:
                termino_df[termino] = 1
    N = len(collection)
    termino_idf = {}
    for termino in termino_df:
        termino_idf[termino] = np.log10(N / termino_df[termino])

    tf_idf = []
    for i in range(len(collection)):
        tf_idf.append([])
        for termino in terminos:
            if termino in collection[i]:
                tf_idf[i].append(np.log10(1 + collection[i][termino]) * termino_idf[termino])
            else:
                tf_idf[i].append(0)
    return tf_idf


def cosine_sim(Q, Doc):
    return np.dot(Q, Doc) / (np.linalg.norm(Q) * np.linalg.norm(Doc))


textos_tfidf = compute_tfidf(textos_procesados)

matriz = []
for doc1 in textos_tfidf:
    row = []
    for doc2 in textos_tfidf:
        row.append(round(cosine_sim(doc1, doc2), 3))
    matriz.append(row)

print(matriz)

"""## P4- Indice invertido con similitud de coseno
### 1- Estructura del índice invertido en Python:

index = {
w1 : [(doc1, tf_w1_doc1), (doc3, tf_w1_doc3),(doc4, tf_w1_doc4),(doc10, tf_w1_doc10)],
w2 : [(doc1, tf_w2_doc1 ), (doc2, tf_w2_doc2)],
w3 : [(doc2, tf_w3_doc2), (doc3, tf_w3_doc3),(doc7, tf_w3_doc7)],
}

idf = {
w1 : idf_w1,
w2 : idf_w2,
w3 : idf_w3,
}

length ={
doc1: norm_doc1,
doc2: norm_doc2,
doc3: norm_doc3,
...
}

### 2- Algoritmo para construir el índice



### 3-	Función de recuperación usando la similitud de coseno:
"""


class InvertIndex:

    def __init__(self, index_file):
        self.index_file = index_file
        self.index = {}
        self.idf = {}
        self.length = {}

    def building(self, collection_text):
        # build the inverted index with the collection
        # compute the tf
        # compute the idf
        # compute the length (norm)
        # store in disk
        pass

    def retrieval(self, query, k):
        self.load_index(self.index_file)
        # diccionario para el score
        score = {}
        # preprocesar la query: extraer los terminos unicos
        query_terms = self.get_terms(query)

        # calcular el tf-idf del query
        tfidf_query = self.get_tfidf(query)

        # aplicar similitud de coseno y guardarlo en el diccionario score
        for term in query_terms:
            list_pub = self.index[term]
            idf = self.idf[term]
            for (docid, tf) in list_pub:
                if docid not in score:
                    score[docid] = 0
                tfidf_doc = tf * idf
                score[docid] += tfidf_query[term] * tfidf_doc

        # norma
        for docid in self.length:
            score[docid] = score[docid] / (self.lenght[docid] * self.get_norm(tfidf_query))

        # ordenar el score de forma descendente
        result = sorted(score.items(), key=lambda tup: tup[1], reverse=True)

        # retornamos los k documentos mas relevantes (de mayor similitud al query)
        return result[:k]
