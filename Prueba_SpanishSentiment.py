# from classifier import SentimentClassifier
from sentiment_analysis_spanish import sentiment_analysis
import pandas as pd


# # clf = SentimentClassifier()

# sentiment = sentiment_analysis.SentimentAnalysisSpanish()

# df = pd.read_excel("Libro1_1_170.xlsx")
# # df = df.dropna()

# # print(df[1:170])

# lista_comentarios = df["Column2"].tolist()

# # print(lista_comentarios)

# for i in range(0, len(lista_comentarios)):
#     print(lista_comentarios[i])
#     print(sentiment.sentiment(lista_comentarios[i]))
#     print("")
#     df.loc[i, 'Spanish Sentiment Analisis'] = sentiment.sentiment(lista_comentarios[i])

#     ## guardar cada prediccion 

# ## escribir en el archivo libro1_1_170.csv sin sobreescribir
# df.to_csv("Libro1_1_170.csv", mode='a3', header=False, index=False)

class SpanishSentimentAnalisis:
    def __init__(self):
        self.clf =  sentiment_analysis.SentimentAnalysisSpanish()
        # pass

    def sentiment(self, text):
        try:
            # sent = sentiment_analysis.SentimentAnalysisSpanish()
            result = self.clf.sentiment(text)
            return result
        except:
            return "Error"



