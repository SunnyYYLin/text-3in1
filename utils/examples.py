from configs import BaseConfig
from models import SentimentModel, Translater, NER_CRF
from data_utils import SentimentDataset, TranslationDataset, NERDataset
from data_utils import SentimentDataCollator, TranslationDataCollator, NERDataCollator
from metrics.ner_metrics import NERMetrics
import random
import torch
import time
from pathlib import Path
import json
from tqdm import tqdm

BATCH_SIZE = 128

def get_examples(model: SentimentModel|Translater|NER_CRF, 
                 dataset: SentimentDataset|TranslationDataset|NERDataset, 
                 config: BaseConfig):
    random.seed(time.time())
    match config.task:
        case 'sentiment':
            return get_sentiment_examples(model, dataset, config)
        case 'translation':
            return get_translation_examples(model, dataset, config)
        case 'ner':
            return get_ner_examples(model, dataset, config)
        case _:
            raise NotImplementedError(f"Task {config.task} not implemented")
        
@torch.no_grad()
def get_sentiment_examples(model: SentimentModel, 
                           dataset: SentimentDataset,
                           config: BaseConfig):
    results = []
    for i in tqdm(range(len(dataset)//BATCH_SIZE)):
        samples = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE] if i < len(dataset)//BATCH_SIZE-1 else dataset[i*BATCH_SIZE:]
        collator = SentimentDataCollator()
        batch = collator(samples)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        outputs = model(input_ids, attention_mask)
        preds = outputs['logits'].argmax(dim=-1)
        results += [{
                'input': dataset.decode_input(input_ids),
                'label': dataset.decode_label(labels),
                'pred': dataset.decode_label(preds)
            } for input_ids, labels, preds in zip(input_ids, labels, preds)]
    wrong_results = [result for result in results if result['label'] != result['pred']]
    if not (Path(__file__).parent.parent/'results').exists():
        (Path(__file__).parent.parent/'results').mkdir()
    with open(Path(__file__).parent.parent/'results'/'sentiment.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(Path(__file__).parent.parent/'results'/'sentiment_wrong.json', 'w') as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=4)
    return 1-len(wrong_results)/len(results)

def get_ner_examples(model: NER_CRF, 
                     dataset: NERDataset, 
                     config: BaseConfig):
    results = []
    metircs = NERMetrics(config)
    for i in tqdm(range(len(dataset)//BATCH_SIZE)):
        samples = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE] if i < len(dataset)//BATCH_SIZE-1 else dataset[i*BATCH_SIZE:]
        collator = NERDataCollator()
        batch = collator(samples)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        outputs = model(input_ids, attention_mask)['logits']
        # print(outputs, attention_mask)
        preds = model.crf.decode(outputs, attention_mask)
        results += [{
                'input': dataset.decode_input(input_ids),
                'label_entities': metircs.extract_entities(dataset.decode_input(input_ids), dataset.decode_label(labels, attention_mask)),
                'pred_entities': metircs.extract_entities(dataset.decode_input(input_ids), dataset.decode_label(preds, attention_mask))
            } for input_ids, labels, preds in zip(input_ids, labels, preds)]
    wrong_results = [result for result in results if result['label_entities'] != result['pred_entities']]
    if not (Path(__file__).parent.parent/'results').exists():
        (Path(__file__).parent.parent/'results').mkdir()
    with open(Path(__file__).parent.parent/'results'/'ner.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    with open(Path(__file__).parent.parent/'results'/'ner_wrong.json', 'w') as f:
        json.dump(wrong_results, f, ensure_ascii=False, indent=4)
    return 1-len(wrong_results)/len(results)

def get_translation_examples(model: Translater, 
                             dataset: TranslationDataset, 
                             config: BaseConfig) -> list[dict[str, any]]:
    results = []
    for i in tqdm(range(len(dataset)//BATCH_SIZE)):
        samples = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE] if i < len(dataset)//BATCH_SIZE-1 else dataset[i*BATCH_SIZE:]
        collator = TranslationDataCollator()
        batch = collator(samples)
        src_ids = batch['src_ids'].to(config.device)
        src_mask = batch['src_padding_mask'].to(config.device)
        tgt_ids = batch['tgt_ids'].to(config.device)
        tgt_mask = batch['tgt_padding_mask'].to(config.device)
        outputs = model(src_ids, src_mask, tgt_ids, tgt_mask)
        preds = outputs['logits'].argmax(dim=-1)
        results += [{
                'src': dataset.decode_src(src_ids[i]),
                'tgt': dataset.decode_tgt(tgt_ids[i]),
                'pred': dataset.decode_tgt(preds[i])
            } for i in range(len(samples))]
    if not (Path(__file__).parent.parent/'results').exists():
        (Path(__file__).parent.parent/'results').mkdir()
    with open(Path(__file__).parent.parent/'results'/'translation.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return len(results)
        