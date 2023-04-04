from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch ## Libreria para el manejo de tensores
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim ## nn es la libreria de redes neuronales y optim es la libreria de optimizacion
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap 


RANDOM_SEED = 42
MAX_LEN = 200 ## Longitud maxima de los comentarios se cortan los comentarios que superen esta longitud, despues cambiar
BATCH_SIZE = 16 ## numero de comentarios que se procesan en cada iteracion
DATASET_PATH = "Libro1.xlsx" ## ruta del dataset
NCLASES = 2 ## numero de clases que tiene el dataset

np.random.seed(RANDOM_SEED) ## para que los resultados sean reproducibles
torch.manual_seed(RANDOM_SEED) ## para inicializar los pesos de la red de forma deterministica

## data set
df = pd.read_excel("Libro1.xlsx")
df = df.dropna()

#eliminar columa1
df = df.drop(columns=['Column1']) ## eliminar column1 text-mining no lo necesito

## eliminar todos los datos que en la columna3 digan sin contexto
df = df.drop(df[df['Column3'] == 'sin contexto'].index)
#eliminar los neutros
df = df.drop(df[df['Column3'] == 'neutro'].index)

lista_comentarios = df["Column2"].tolist()

# cambiio los valores de la columna3 a 1 y 0
df['Column3'] = df['Column3'].replace(['positivo'], 1)
df['Column3'] = df['Column3'].replace(['negativo'], 0)
# print(df)

# tokenizar los comentarios
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


#EJEMPLO DE TOKENIZACION
SAMPLE_TEXT = 'Me encanta el nuevo celular'
tokens = tokenizer.tokenize(SAMPLE_TEXT)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f' Sentence: {SAMPLE_TEXT}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')


#Codificacion para bert
encoding = tokenizer.encode_plus(
    SAMPLE_TEXT,
    max_length=200,
    truncation=True,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    pad_to_max_length=True, # agregar ceros hasta llegar a la longitud maxima
    return_attention_mask=True,  # hace que los ceros no sean tomados en cuenta
    return_tensors='pt',  # Return PyTorch tensors
)
encoding.keys()
#  diccionario que contiene los tokens, la mascara (de 0 y 1).  

class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews #    lista de comentarios
        self.targets = targets # si es positivo o negativo
        self.tokenizer = tokenizer # tokenizer
        self.max_len = max_len # longitud maxima de los comentarios

    def __len__(self): 
        # longitud del dataset
        return len(self.reviews)
    
    def __getitem__(self, item):
        # obtener un elemento del dataset
        review = str(self.reviews[item]) # toma el comentario en la posicion item
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True, # agregar ceros hasta llegar a la longitud maxima
            return_attention_mask=True,  # hace que los ceros no sean tomados en cuenta
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long) # si es positivo o negativo
        }

# data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = IMDBDataset(
        reviews = df["Column2"].to_numpy(), # lista de comentarios 
        targets = df["Column3"].to_numpy(), # lista de etiquetas
        tokenizer =  tokenizer,
        max_len = MAX_LEN
    )
    return DataLoader( dataset, batch_size = BATCH_SIZE, num_workers = 4)

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED) # 90% train y 10% test
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE) # entrenamiento
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE) # validacion

# modelo
class SentimentClassifier(nn.Module):  # hereda de nn.Module que es una red neuronal de pytorch
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__() # inicializa la clase padre
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3) # 30% de neuronas se apagan en cada iteracion para evitar el overfitting
        self.linearOut = nn.Linear(self.bert.config.hidden_size, n_classes) # bert.config.hidden_size = 768 (dimension de la salida de bert), n_classes = 2 (positivo o negativo)
       
    
    def forward(self, input_ids, attention_mask): # input_ids = tokens, attention_mask = mascara de 0 y 1
        _, cls_output = self.bert(  # cls_output = salida de la ultima capa de bert, la primera es _ (no se usa)
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        drop_output = self.drop(cls_output) # salida de la capa de dropout
        return self.linearOut(drop_output) # salida de la capa lineal


model = SentimentClassifier(2) # 2 clases (positivo o negativo)
#entrenar el modelo
# model = model.to(device) # enviar el modelo a la gpu
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) # optimizador AdamW, lr = learning rate 
total_steps = len(train_data_loader) * EPOCHS # numero de iteraciones
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) # scheduler para el learning rate
loss_fn = nn.CrossEntropyLoss()


def train_model(model, data_loader, loss_fn, optimizer,scheduler, n_examples):
    model = model.train() # poner el modelo en modo entrenamiento
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"] # tokens
        attention_mask = d["attention_mask"]# mascara de 0 y 1
        targets = d["targets"] # etiquetas
        outputs = model(input_ids=input_ids, attention_mask=attention_mask) # salida del modelo
        _, preds = torch.max(outputs, dim=1) # obtener la clase con mayor probabilidad
        loss = loss_fn(outputs, targets) # calcular la perdida
        correct_predictions += torch.sum(preds == targets) # calcular la precision
        losses.append(loss.item()) # agregar la perdida a la lista
        loss.backward() # retropropagacion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # evitar que los gradientes exploten
        optimizer.step() # actualizar los pesos
        scheduler.step() # actualizar el learning rate
        optimizer.zero_grad() # reiniciar los gradientes
    return correct_predictions.double() / n_examples, np.mean(losses) # precision y perdida promedio


def test_model(model,data_loader, loss_fn, n_examples):
    model = model.eval() # poner el modelo en modo evaluacion
    losses = []
    correct_predictions = 0
    with torch.no_grad(): # no se calculan los gradientes
        for d in data_loader:
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    test_acc, test_loss = test_model(model, test_data_loader, loss_fn, len(df_test))
    print(f'Test loss {test_loss} accuracy {test_acc}')
    print()


def Classification(text):
    encoding_review = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding_review['input_ids']
    attention_mask = encoding_review['attention_mask']
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    return prediction

# probando el modelo

review_text = "Los 2 doctores de medicina general que atendieron a mi madre, fueron muy claros, examinaron bien, muy amables..."
Classification(review_text)





















































# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# # model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=1)

# RANDOM_SEED = 42
# MAX_LEN = 200 ## Longitud maxima de los comentarios se cortan los comentarios que superen esta longitud, despues cambiar

# ## data set
# df = pd.read_excel("Libro1.xlsx")
# df = df.dropna()
# lista_comentarios = df["Column2"].tolist()

# comentarios = lista_comentarios[1:25]

# np.random.seed(RANDOM_SEED) ## para que los resultados sean reproducibles
# torch.manual_seed(RANDOM_SEED) ## para inicializar los pesos de la red de forma deterministica

# for comentario in comentarios:
#     print(comentario)
#     print("")

 
    
        