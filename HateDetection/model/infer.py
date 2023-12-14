import re
import emoji
import argparse

import torch
from torch.nn import functional as F
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from soynlp.normalizer import repeat_normalize

# GPU 사용
device = torch.device("cuda")

parser = argparse.ArgumentParser(description='Hate Detection')

parser.add_argument('--model-path',
                    type=str,
                    default=True,
                    help='model pt file')

args = parser.parse_args()

model_path = "/content/drive/MyDrive/KEA_dataset/hateDetectionmodel.pt"

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)


def infer(model, x, path) :
    model.load_state_dict(torch.load(path))

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )
    processed = pattern.sub(' ', x)
    processed = url_pattern.sub(' ', processed)
    processed = processed.strip()
    processed = repeat_normalize(processed, num_repeats=2)

    tokenized = tokenizer(processed, return_tensors='pt')
    tokenized = tokenized.to(model.device)

    output = model(tokenized.input_ids, tokenized.attention_mask)
    sm = F.softmax(output.logits, dim=-1)
    result = np.argmax(sm)

    return result


text = input()
print(infer(model, text, model_path))
# 0 -> non-hate
# 1 -> hate
