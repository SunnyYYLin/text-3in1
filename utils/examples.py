from configs import BaseConfig
from models import SentimentModel, Translater, NER_CRF
from data_utils import SentimentDataset, TranslationDataset, NERDataset
from data_utils import SentimentDataCollator, TranslationDataCollator, NERDataCollator
import random
import torch
import time

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
        
def get_sentiment_examples(model: SentimentModel, 
                           dataset: SentimentDataset,
                           config: BaseConfig) -> list[dict[str, any]]:
    samples = random.choices(dataset, k=3)
    collator = SentimentDataCollator()
    batch = collator(samples)
    input_ids = batch['input_ids'].to(config.device)
    attention_mask = batch['attention_mask'].to(config.device)
    labels = batch['labels'].to(config.device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = outputs['logits'].argmax(dim=-1)
    examples = []
    for i in range(3):
        example = {
            'input': dataset.decode_input(input_ids[i]),
            'label': dataset.decode_label(labels[i]),
            'pred': dataset.decode_label(preds[i])
        }
        examples.append(example)
    return examples

def get_ner_examples(model: NER_CRF, 
                     dataset: NERDataset, 
                     config: BaseConfig) -> list[dict[str, any]]:
    samples = random.choices(dataset, k=3)
    collator = NERDataCollator()
    batch = collator(samples)
    input_ids = batch['input_ids'].to(config.device)
    attention_mask = batch['attention_mask'].to(config.device)
    labels = batch['labels'].to(config.device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = model.crf.decode(outputs['logits'], attention_mask)
    examples = []
    for i in range(3):
        example = {
            'input': dataset.decode_input(input_ids[i]),
            'pred': dataset.decode_label(preds[i], attention_mask[i]),
            'label': dataset.decode_label(labels[i], attention_mask[i])
        }
        examples.append(example)
    return examples

def get_translation_examples(model: Translater, 
                             dataset: TranslationDataset, 
                             config: BaseConfig) -> list[dict[str, any]]:
    samples = random.choices(dataset, k=3)
    collator = TranslationDataCollator()
    batch = collator(samples)
    src_ids = batch['src_ids'].to(config.device)
    src_mask = batch['src_padding_mask'].to(config.device)
    tgt_ids = batch['tgt_ids'].to(config.device)
    tgt_mask = batch['tgt_padding_mask'].to(config.device)
    with torch.no_grad():
        outputs = model(src_ids, src_mask, tgt_ids, tgt_mask)
        preds = outputs['logits'].argmax(dim=-1)
    examples = []
    for i in range(3):
        example = {
            'src': dataset.decode_src(src_ids[i], src_mask[i]),
            'tgt': dataset.decode_tgt(tgt_ids[i], tgt_mask[i]),
            'pred': dataset.decode_tgt(preds[i], tgt_mask[i])
        }
        examples.append(example)
    return examples
        