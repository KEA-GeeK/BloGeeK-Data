from kobert_tokenizer import KoBERTTokenizer
import torch
from torch import nn
import pickle 
from torch.utils.data import Dataset
import gluonnlp as nlp
from gluonnlp import vocab as voc
import numpy as np
from transformers import BertModel
import argparse


device = torch.device("cuda:0")
max_len = 64
batch_size = 64

parser = argparse.ArgumentParser(description='Porarity Recognition Model')

parser.add_argument('--pt_path',
                    type=str,
                    default=True,
                    help='location of pt file')

args = parser.parse_args()

pt_path = args.pt_path

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tok=tokenizer.tokenize
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

parser = argparse.ArgumentParser(description='Porarity Recognition Model')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load(pt_path))

def predict(predict_sentence):

  data = [predict_sentence, '0']
  dataset_another = [data]

  another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
  model.eval()

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)

    valid_length= valid_length
    label = label.long().to(device)

    out = model(token_ids, valid_length, segment_ids)


    test_eval=[]
    for i in out:
      logits=i
      logits = logits.detach().cpu().numpy()

      if np.argmax(logits) == 0 :
        polarity = "negative"

      elif np.argmax(logits) == 1 :
        polarity = "positive"
      
      elif np.argmax(logits) == 2 :
        polarity = "neutral"

      else :
        print("Error")

      test_eval.append(polarity)
  return test_eval[0]

#중지 코드 = EXIT THIS
while True :
    sentence = input()
    if sentence == "EXIT THIS" :
        break
    print(predict(sentence))
    print("\n")