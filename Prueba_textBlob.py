# import pandas as pd
from textblob import TextBlob
# df = pd.read_excel("Libro1_1_170.xlsx")
# lista_comentarios = df["Column2"].tolist()

# for i in range(0, len(lista_comentarios)):
#     ## traducir a ingles 

#     comentario = TextBlob(lista_comentarios[i]).translate(from_lang='es', to='en')
#     result = comentario.sentiment.polarity

#    ## devuelve un valor entre -1 y 1, pasar a valores entre 0 y 1
#     result = (result + 1) / 2

class TextBlobSentiment:
    def __init__(self):
        pass

    def sentiment(self, text):
        try:
            analysis = TextBlob(text)
            result = analysis.sentiment.polarity
            result = (result + 1) / 2
            if result>0.6:
                return "positivo"
            elif result<0.4:
                return "negativo"
            else:
                return "neutral"
        except:
            return "Error"