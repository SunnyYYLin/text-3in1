from torch.utils.data import DataLoader
from models.text_cnn import TextCNN
from utils.train import train, evaluate
from dataset import SentimentDataset, collate_fn
import json
import argparse
from collections import namedtuple
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='checkpoints')
    parser.add_argument('--train', default='./data/train.jsonl')
    parser.add_argument('--test', default='./data/test.jsonl')
    parser.add_argument('--val', default='./data/val.jsonl')
    parser.add_argument('--num_epoch', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--vocab', default='./data/vocab.json')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--filter_sizes', default=[4,4,5,5,6,6], nargs='+', type=int)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--verbose', default=True, type=bool)
    arg = parser.parse_args()
    
    if not os.path.exists(arg.save_path):
        os.makedirs(arg.save_path)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    
    filter_config_name = '-'.join([str(size) for size in arg.filter_sizes])
    config_name: str = '_'.join([filter_config_name,
                           str(arg.hidden_dim),
                           f"{arg.dropout*100:.0f}", 
                           f"{arg.lr:.0e}"])
    config_name = 'TextCNN_' + config_name
    arg.save_path = os.path.join(arg.save_path, config_name)
    arg.log_dir = os.path.join(arg.log_dir, config_name)
    print(arg)
    
    train_dataset = SentimentDataset(arg.train, arg.vocab)
    val_dataset = SentimentDataset(arg.val, arg.vocab)
    test_dataset = SentimentDataset(arg.test, arg.vocab)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    with open(arg.vocab, 'rb') as f:
        chr_vocab = json.load(f)
    config = {
        'dropout': arg.dropout,
        'num_classes': 2,
        'vocab_size': len(chr_vocab),
        'embedding_dim':arg.hidden_dim,
        'filter_sizes':arg.filter_sizes
    }
    config = namedtuple('config', config.keys())(**config)
    model = TextCNN(config)
    train(model, train_loader, val_loader, arg)