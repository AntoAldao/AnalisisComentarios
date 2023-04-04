import pandas as pd
from Prueba_textBlob import TextBlobSentiment
from Prueba_chatgpt import ChatGPTSentiment
from Prueba_SpanishSentiment import SpanishSentimentAnalisis
import time

## leer archivo libro 1.xlsx
df = pd.read_excel("Libro1.xlsx")
df = df.dropna()

## datos del 1 al 170 

df = df.drop(columns=['Column1']) ## eliminar column1 text-mining no lo necesito 
print(df[1:170])

## eliminar todos los datos que en la columna3 no digan sin contexto
# df = df.drop(df[df['Colum+n3'] == 'sin contexto'].index)

## agrego clumna chatgpt, Spanish Sentiment Analisis, BERT, textBlob con valor None 
df["chatgpt"] = None
df["Spanish Sentiment Analisis"] = None
df["BERT"] = None
df["textBlob"] = None

lista_comentarios = df["Column2"].tolist()
print(len(lista_comentarios))

# # comentarios = lista_comentarios[1:170]
chatGPT = ChatGPTSentiment()
spanishtAnalisis = SpanishSentimentAnalisis()
textblob = TextBlobSentiment()
for i in range(0, len(lista_comentarios)):
    print(i)
    # print(lista_comentarios[i])
    # print("")
    # print("chatgpt")
    # print(ChatGPTSentiment.sentiment(lista_comentarios[i]))
    # print("")
    # print("Spanish Sentiment Analisis")
    # print(Prueba_SpanishSentiment.sentiment(lista_comentarios[i]))
    # print("")
    # print("textBlob")
    # print(TextBlobSentiment.sentiment(text=lista_comentarios[i]))
    # print("")
    # print("")
    
    # # print(ChatGPT.sentiment(lista_comentarios[i]))
    df["chatgpt"][i] = chatGPT.sentiment(lista_comentarios[i])
    df["Spanish Sentiment Analisis"][i] =  spanishtAnalisis.sentiment(lista_comentarios[i])
    # print(spanishtAnalisis.sentiment(lista_comentarios[i]))
    df["textBlob"][i] = textblob.sentiment(lista_comentarios[i])
    time.sleep(2) ## esperar 2 segundos porque chatgpt tiene un limite de 20 peticiones por minuto



# escribir en el archivo libro1_1_170.xlsx 
df.to_excel("libro1_1_170.xlsx")

