from transformers import GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
import pickle


tokenizer = GPT2Tokenizer.from_pretrained('/home/zengrui/zengrui/gpt2')
df = pd.read_csv('/data/zengrui/nlp_dataset/wiki-dataset/wikitext-103-v1-train.csv')
text_list = [df['text'][i] for i in range(df.shape[0])]
batch_size = 128
iteration = len(text_list) // batch_size
encoding = {'input_ids': [], 'attention_mask': []}
for i in tqdm(range(iteration)):
    batch_texts = text_list[i * batch_size: (i + 1) * batch_size]
    batch_encoding = tokenizer(batch_texts, truncation=True)
    for key, value in batch_encoding.items():
        encoding[key].extend(value)
with open('/data/zengrui/nlp_dataset/wiki-dataset/wikitext-1-3-v1-train-gpt2-encoding.pkl', 'wb') as f:
    pickle.dump(encoding, f)
