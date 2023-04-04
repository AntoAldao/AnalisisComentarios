from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup

import torch ## Libreria para el manejo de tensores
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim ## nn es la libreria de redes neuronales y optim es la libreria de optimizacion
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap 

# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# # model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=1)

RANDOM_SEED = 42
MAX_LEN = 200 ## Longitud maxima de los comentarios se cortan los comentarios que superen esta longitud, despues cambiar

## data set
df = pd.read_excel("Libro1.xlsx")
df = df.dropna()
lista_comentarios = df["Column2"].tolist()

comentarios = lista_comentarios[1:25]

np.random.seed(RANDOM_SEED) ## para que los resultados sean reproducibles
torch.manual_seed(RANDOM_SEED) ## para inicializar los pesos de la red de forma deterministica

for comentario in comentarios:
    print(comentario)
    print("")

 
    
        