from tqdm import tqdm
import json
from models.vocab import Vocabulary

# build vocab
with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    tokenized_sentences = [json.loads(line)['text'] for line in f.readlines()]

words = []
for sentence in tokenized_sentences:
    words.extend(sentence)
    
vocab = Vocabulary(words, max_vocab=5000)
vocab.save('data/vocab.pth')