def infer(x, path) :
    model.load_state_dict(torch.load(path))
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    
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
    return F.softmax(output.logits, dim=-1)

path = "/content/drive/MyDrive/KEA_dataset/hate_epc10"
text = "sentence"
print(infer(text, path))