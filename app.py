# criar programa bara analisar o arquivo pdf ou txt fornecido pelo usuario
# e retornar o resultado da analise
# importar bibliotecas
import streamlit as st
import pandas as pd
import plotly.express as px
import pdfplumber
import re
import time
import os
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googletrans import Translator
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

nltk.download('all')


# função para pedir o arquivo ao usuario
def get_file():
    try:
        uploaded_file = st.file_uploader("Choose a file")
        if st.button('Upload'):
            if uploaded_file is not None:
                with open("arquivo.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # mostra barra de processo de carregamento
                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('Upload realizado com sucesso!')
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível carregar o arquivo")


def read_file_pdf():
    # ler o arquivo pdf
    try:
        with pdfplumber.open("arquivo.pdf") as pdf:
            # mostrar o texto do arquivo
            texto = ""
            for page in pdf.pages:
                texto += page.extract_text()
            return texto
    except Exception as e:
        st.error(e)
        st.error("Não foi possível ler o arquivo")


# tratar o texto usando o regex
def limpar_texto(texto):
    try:
        texto = texto.lower()
        # remover pontuação
        texto = re.sub(r'[^\w\s]', '', texto)
        # remover numeros
        texto = re.sub(r'[0-9]', '', texto)
        texto = re.sub(r'\d+', '', texto)
        # remover espaços em branco
        texto = texto.strip()
        # remover caracteres especiais
        texto = re.sub(r'[^\w\s]', '', texto)
        # remover acentos
        texto = re.sub(r'[áàâã]', 'a', texto)
        texto = re.sub(r'[éèê]', 'e', texto)
        texto = re.sub(r'[íì]', 'i', texto)
        texto = re.sub(r'[óòôõ]', 'o', texto)
        texto = re.sub(r'[úù]', 'u', texto)
        texto = re.sub(r'[ç]', 'c', texto)
        # remover aspas simples
        texto = re.sub(r'[’]', '', texto)
        # remover aspas duplas
        texto = re.sub(r'["]', '', texto)
        # remover letras soltas
        texto = re.sub(r'\b\w\b', '', texto)
        # remover algarismos romanos
        texto = re.sub(r'\b[ivx]+\b', '', texto)
        # remover links
        texto = re.sub(r'http\S+', '', texto)
        # remover palavras com igual ou menos de 3 letras
        texto = re.sub(r'\b\w{1,3}\b', '', texto)
        return texto
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível limpar o texto")


# remover as usando o nltk
def remover_stop_words(texto):
    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        palavras = word_tokenize(texto)
        palavras_filtradas = []
        for w in palavras:
            if w not in stop_words:
                palavras_filtradas.append(w)
        return " ".join(palavras_filtradas)
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível remover as stop words")


def retorna_texto():
    try:
        if os.path.exists("arquivo.pdf"):
            texto = read_file_pdf()
            texto = limpar_texto(texto)
            texto = remover_stop_words(texto)
            return texto
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível ler o arquivo")


def retorna_puro():
    try:
        if os.path.exists("arquivo.pdf"):
            texto = read_file_pdf()
            texto = remover_stop_words(texto)
            # remover numeração de páginas
            texto = re.sub(r'[0-9]', '', texto)
            texto = re.sub(r'\d+', '', texto)
            return texto
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível ler o arquivo")


# mostar dataframe com as palavras e a quantidade de vezes que aparecem
def mostra_df():
    try:
        dados = retorna_texto()
        dados = dados.split()
        dados = pd.DataFrame(dados, columns=["Palavras"])
        dados = dados["Palavras"].value_counts()
        dados = pd.DataFrame(dados)
        dados = dados.rename(columns={"Palavras": "Quantidade"})
        dados = dados.reset_index()
        dados = dados.rename(columns={"index": "Palavras"})
        return dados
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível gerar o dataframe")


# mostrar o grafico de barras
def mostra_grafico_barras(dataframe):
    # criar grafico de barras somente com top 15 palavras
    try:
        dados = dataframe.head(15)
        fig = px.bar(dados, x="Palavras", y="Quantidade", color="Quantidade", title="Top 15 palavras")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível gerar o gráfico de barras")


# mostrar o grafico de pizza
def mostra_grafico_pizza(dataframe):
    # criar grafico de pizza somente com top 5 palavras
    try:
        dados = dataframe.head(5)
        fig = px.pie(dados, values="Quantidade", names="Palavras", title="Top 5 palavras")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível gerar o gráfico de pizza")


# criar wordcloud com as palavras
def mostra_grafico_nuvem():
    try:
        texto = retorna_texto()
        # remover palavras duplicadas
        texto = set(texto.split())
        texto = " ".join(texto)
        # remover palavras com menos de 3 letras
        texto = re.sub(r'\b\w{1,3}\b', '', texto)
        # remover palavras com mais de 15 letras
        texto = re.sub(r'\b\w{15,}\b', '', texto)
        wordcloud = WordCloud(width=800, height=500, max_font_size=110).generate(texto)
        st.image(wordcloud.to_array())
    except Exception as e:
        st.error(e)
        st.warning("Não foi possível gerar a nuvem de palavras")


# analise de sentimento da pagina escolhida pelo usuario
def analise_sentimento():
    try:
        nltk.download('vader_lexicon')
        arquivo = retorna_texto()
        # pergunta ao usuario qual pagina ele quer analisar com valor padrao 1 minimo 1
        pagina = st.number_input("Qual página você quer analisar?", min_value=0, value=0)
        # mostrar o texto da pagina escolhida
        if os.path.exists("arquivo.pdf"):
            with pdfplumber.open("arquivo.pdf") as pdf:
                texto = pdf.pages[pagina].extract_text()
                st.markdown("""<div style='text-align: justify;'>{}</div>""".format(texto), unsafe_allow_html=True)
                # se o texo estiver em portugues, traduzir para ingles
                translator = Translator()
                texto = translator.translate(texto, dest="en").text
                # analise de sentimento
                sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(texto)
                # mostrar o resultado da analise de sentimento usando  st.markdown
                st.markdown("""<div style='text-align: justify;'>{}</div>""".format(ss), unsafe_allow_html=True)
                if ss["compound"] >= 0.05:
                    st.success("O Sentimento dessa pagina e positivo")
                elif ss["compound"] <= -0.05:
                    st.error("O Sentimento dessa pagina Negativo")
                else:
                    st.warning("O Sentimento dessa pagina Neutro")
    except Exception as e:
        st.error(e)
        st.warning("Erro ao analisar o sentimento")


def resumo_geral(text, per):
    try:
        # traduzir o texto para ingles
        translator = Translator()
        text = translator.translate(text, dest="en").text
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        tokens = [token.text for token in doc]
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        select_length = int(len(sentence_tokens) * per)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ''.join(final_summary)
        return summary
    except Exception as e:
        st.error(e)
        st.warning("Erro ao gerar o resumo")


def pegar_texto_pagina():
    try:
        # pergunta ao usuario qual pagina ele quer analisar com valor padrao 1 minimo 1
        pagina = st.number_input("Qual página você quer analisar?", min_value=0, value=0)
        # mostrar o texto da pagina escolhida
        if os.path.exists("arquivo.pdf"):
            with pdfplumber.open("arquivo.pdf") as pdf:
                texto = pdf.pages[pagina].extract_text()
                # remover numeracao das paginas
                texto = re.sub(r'\d+', '', texto)
                st.markdown("""<div style='text-align: justify;'>{}</div>""".format(texto), unsafe_allow_html=True)
                return texto
    except Exception as e:
        st.error(e)
        st.warning("Erro ao pegar o texto da pagina")


def mostrar_texto_original():
    if os.path.exists("arquivo.pdf"):
        try:
            with pdfplumber.open("arquivo.pdf") as pdf:
                # mostra a quantidade de paginas do arquivo
                st.write("O arquivo PDF tem {}".format(len(pdf.pages)))
                # pergunta ao usuario qual pagina ele quer analisar com valor padrao 0 minimo 0
                pagina = st.number_input("Qual página você quer analisar?", min_value=0, value=0)
                # mostrar o texto da pagina escolhida
                texto = pdf.pages[pagina].extract_text()
                # remover numeracao das paginas
                texto = re.sub(r'\d+', '', texto)
                # criar botao para mostrar o texto
                if st.button("Mostrar Texto"):
                    # mostrar o texto da pagina escolhida justificado
                    st.markdown("""<div style='text-align: justify;'>{}</div>""".format(texto), unsafe_allow_html=True)
        except Exception as e:
            st.error(e)
            st.warning("Erro ao mostrar o texto")


# função para configurar paginas
def configurar_paginas():
    # deixar o menu fixo
    st.set_page_config(page_title="Analise de PDF", page_icon=":page_facing_up:", initial_sidebar_state="expanded")
    # deixa botoes com cores diferentes
    st.markdown(
        """
    <style>
    .reportview-container .main .block-container{{
        max-width: 1000px;
        padding-top: 10px;
        padding-right: 10px;
        padding-left: 10px;
        padding-bottom: 10px;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    # configurar paginas
    configurar_paginas()
    # criar menu
    menu = ["Upload", "Mostrar Texto original", "Mostrar Texto tratado", "Mostrar DataFrame", "Mostrar Gráfico Barras",
            "Mostrar Gráfico Pizza", "Analise de Sentimento", "wordcloud", "Resumo por Pagina"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Upload":
        st.title("Upload de arquivo")
        st.subheader("Por favor faça upload do arquivo PDF")
        get_file()
    elif choice == "Mostrar Texto original":
        st.title("Trecho do Texto original")
        st.subheader("Mostrar Thecho do texto original")
        mostrar_texto_original()

    elif choice == "Mostrar Texto tratado":
        st.markdown("<h1 style='text-align: center; color: white;'>Trecho Tratado</h1>", unsafe_allow_html=True)
        texto = retorna_texto()
        # mostra apenas 20% do texto
        st.markdown("""<div style='text-align: justify;'>{}</div>""".format(texto[:int(len(texto) * 0.2)]),
                    unsafe_allow_html=True)

    elif choice == "Mostrar DataFrame":
        st.markdown("<h1 style='text-align: center; color: white;'>Tabela</h1>", unsafe_allow_html=True)
        dataframe = mostra_df()
        # mostrar dataframe alinhado ao centro
        st.table(dataframe.style.set_properties(**{'text-align': 'center'}))

    elif choice == "Mostrar Gráfico Barras":
        st.markdown("<h1 style='text-align: center; color: white;'>Grafico Barra</h1>", unsafe_allow_html=True)
        dataframe = mostra_df()
        mostra_grafico_barras(dataframe)

    elif choice == "Mostrar Gráfico Pizza":
        st.markdown("<h1 style='text-align: center; color: white;'>Grafico Pizza</h1>", unsafe_allow_html=True)
        dataframe = mostra_df()
        mostra_grafico_pizza(dataframe)

    elif choice == "Analise de Sentimento":
        # criar um h1 no streamlit com o titulo Analise de Sentimento alinha ao centro
        st.markdown("<h1 style='text-align: center; color: white;'>Analise de Sentimento</h1>", unsafe_allow_html=True)
        analise_sentimento()

    elif choice == "wordcloud":
        st.markdown("<h1 style='text-align: center; color: white;'>Nuvem de palavras</h1>", unsafe_allow_html=True)
        dataframe = mostra_df()
        mostra_grafico_nuvem()

    elif choice == "Resumo por Pagina":
        st.markdown("<h1 style='text-align: center; color: white;'>Resumo por Pagina</h1>", unsafe_allow_html=True)
        # criar botao para gerar o resumo
        texto = pegar_texto_pagina()
        per = st.slider("Selecione a porcentagem do resumo", 0.05, 1.0)
        if st.button("Gerar Resumo"):
            translator = Translator()
            resumo = resumo_geral(texto, per)
            resumo = translator.translate(resumo, dest="pt").text
            st.markdown("""<div style='text-align: justify;'>{}</div>""".format(resumo), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
