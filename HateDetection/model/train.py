import os
import re
import datetime
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from soynlp.normalizer import repeat_normalize

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm.notebook import tqdm


# GPU 사용
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Hate Detection')

parser.add_argument('--train_data',
                    type=str,
                    default=True,
                    help='train data')

parser.add_argument('--test_data',
                    type=str,
                    default=True,
                    help='test data')

parser.add_argument('--num_epoch',
                    type=str,
                    default=True,
                    help='the number of epoch')

args = parser.parse_args()

training_file_path = args.train_data
test_file_path = args.test_data

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
electramodel = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)



class HateDataset(Dataset):

  def __init__(self, csv_file):
    self.dataset = pd.read_csv(csv_file, sep='\t').dropna(axis=0)
    self.dataset.drop_duplicates(subset=['document'], inplace=True)

    print(self.dataset.describe())

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    row = self.dataset.iloc[idx, 0:2].values
    text = row[0]
    y = row[1]

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        pad_to_max_length=True,
        add_special_tokens=True
        )

    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]

    return input_ids, attention_mask, y
  
  
#데이터셋 처리
train_dataset = HateDataset('training_file_path')
test_dataset = HateDataset('test_file_path')


epochs = args.num_epoch
batch_size = 16

optimizer = AdamW(electramodel.parameters(), lr=5e-6)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


losses = []
accuracies = []

for i in range(epochs):
  total_loss = 0.0
  correct = 0
  total = 0
  batches = 0

  electramodel.train()

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
    optimizer.zero_grad()
    y_batch = y_batch.to(device)
    y_pred = electramodel(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    loss = F.cross_entropy(y_pred, y_batch)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    _, predicted = torch.max(y_pred, 1)
    correct += (predicted == y_batch).sum()
    total += len(y_batch)

    batches += 1
    if batches % 100 == 0:
      print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

  losses.append(total_loss)
  accuracies.append(correct.float() / total)
  print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)


#모델 테스트
electramodel.eval()

test_correct = 0
test_total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
  y_batch = y_batch.to(device)
  y_pred = electramodel(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
  _, predicted = torch.max(y_pred, 1)
  test_correct += (predicted == y_batch).sum()
  test_total += len(y_batch)

print("Accuracy:", test_correct.float() / test_total)



#모델 저장
folder_path = os.path.join(os.path.dirname(__file__), '.', 'pt')

current_date = datetime.date.today()
pt_name = "model_" + current_date.strftime("%Y%m%d") + ".pt"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

pt_path = os.path.join(folder_path, pt_name)

torch.save(electramodel.state_dict(), pt_path)
