#%%
from collections import defaultdict
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import os
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import tensorflow as tf
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers.models import bert
#torch.cuda.empty_cache()
RANDOM_SEED = 63
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%
#* Data input
df_train=pd.read_csv("../../data/train.csv")
df_test=pd.read_csv("../../data/test.csv")
df_train = df_train.drop(['location'],axis=1)
df_test = df_test.drop(['location'],axis=1)
#%%
sns.countplot(df_train.target)
plt.grid()
plt.show()
# There is class imbalance
# %%
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#%%
#* Choosing MAX_LEN
token_lens=[]
for txt in df_train.text:
    tokens = tokenizer.encode(txt, max_length=512,truncation=True)
    token_lens.append(len(tokens))
sns.distplot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()
#%%
MAX_LEN=100
#%%
from sklearn.model_selection import train_test_split
class TweetDataset(Dataset):

    def __init__(self,texts, targets, tokenizer, max_len) -> None:
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = str(self.texts[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = TweetDataset(
    texts=df.text.to_numpy(),
    targets=df.target.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )
#%%
#train,test = train_test_split(df_train,test_size=0.1,random_state=RANDOM_SEED)
#val,test = train_test_split(test,test_size=0.5,random_state=RANDOM_SEED)
BATCH_SIZE=8
#%%
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
#val_data_loader = create_data_loader(val,tokenizer,MAX_LEN,BATCH_SIZE)
#test_data_loader = create_data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)
# %%
data = next(iter(train_data_loader))
# %%
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
# %%
#* Classifier Model
class TweetClassifier(nn.Module):

    def __init__(self, n_classes):
        super(TweetClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(0.5) #* each hidden unit will be set to 0 (dropped out) with probability 0.3.
        #* Done to stochaisticly avoid overfittiny. In a way crippling the NN
        #* Hidden units now cannot co-adapt to other units
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) #* Linear regression using nn

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).to_tuple()
        output = self.drop(pooled_output)
        return self.out(output)

model = TweetClassifier(2)
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape)
print(attention_mask.shape)
#%%
nn.functional.softmax(model(input_ids, attention_mask), dim=1)
# %%
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) #!
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)
# %%
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    model=model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids=d['input_ids'].to(device)
        attention_mask=d['attention_mask'].to(device)
        targets=d['targets'].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs,targets)

        correct_predictions+= torch.sum(preds==targets)
        losses.append(loss.item())


        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0) #! Gradient clipping
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double()/ n_examples, np.mean(losses)
# %%
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids=d['input_ids'].to(device)
            attention_mask=d['attention_mask'].to(device)
            targets=d['targets'].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs,targets)

        correct_predictions+= torch.sum(preds==targets)
        losses.append(loss.item())
    
    return correct_predictions.double()/ n_examples, np.mean(losses)
# %%
from collections import defaultdict
torch.cuda.empty_cache()
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    '''
    val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(val)
    ) 

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()'''
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    #history['val_acc'].append(val_acc)
    #history['val_loss'].append(val_loss)

    if train_acc>best_accuracy:
        torch.save(model.state_dict(),'best_model_state.bin')
        best_accuracy = train_acc
#%%
plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
# %%
#*  prediction
encoded_test_text=df_test.text.apply(lambda x:
  tokenizer.encode_plus(
  x,
  max_length=MAX_LEN,
  add_special_tokens=True,
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
  truncation=True
  ))
input_ids_test = [x['input_ids'] for x in encoded_test_text]
attention_mask_test = [x['attention_mask'] for x in encoded_test_text]
# %%
predictions = []
for x, y in zip(input_ids_test,attention_mask_test):
    x = x.to(device)
    y = y.to(device)
    output = model(x,y)
    _, prediction = torch.max(output, dim=1)
    predictions.append(prediction)
#%%
predictions = pd.Series([p.item() for p in predictions])
submission = pd.concat([df_test.id,predictions],axis=1)
submission.rename(columns = {0:'target'}, inplace=True)
# %%
submission.to_csv('../submission2.csv',index=False)