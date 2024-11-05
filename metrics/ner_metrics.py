import os
import json
from dataclasses import dataclass
import torch
from configs import BaseConfig
from torch.nn import functional as F

@dataclass(frozen=True)
class NamedEntity:
    start: int
    end: int
    type: str

class NERMetrics:
    def __init__(self, config: BaseConfig):
        with open(os.path.join(config.data_path, 'tag_vocab.json'), 'r') as f:
            self.tag2id: dict[str, int] = json.load(f)
        self.id2tag: dict[int, str] = {id: tag for tag, id in self.tag2id.items()}
        self.entity_types = self._get_entity_types()
        self.device = config.device

    def _get_entity_types(self) -> list[str]:
        """
        提取标签词汇表中的所有实体类型，排除 'O' 标签。
        """
        entity_types = set()
        for tag in self.tag2id:
            if tag != "O":
                parts = tag.split('-')
                if len(parts) == 2:
                    entity_types.add(parts[1])
        return list(entity_types)

    def tags2entities(self, tags: list[str]) -> list[NamedEntity]:
        """
        使用 BIO 标注方案从标签序列中提取实体。

        Args:
            tags (list[str]): 标签序列。
        """
        entities = []
        type = None
        start = None

        for idx, tag in enumerate(tags):
            if tag == "O":
                if type is not None:
                    entities.append(NamedEntity(start=start, end=idx, type=type))
                    type = None
            else:
                action, _, this_type = tag.partition('-')
                if action == "B":
                    if type is not None:
                        if type != type:
                            entities.append(NamedEntity(start=start, end=idx, type=type))
                    start = idx
                    type = this_type
                elif action == "I":
                    if type is None:
                        start = idx
                        type = this_type
                else:
                    raise ValueError(f"Invalid tag: {tag}")

        return entities
    
    def ids2tags(self, idxs: list[int]) -> list[str]:
        """
        将标签 ID 转换为标签。

        Args:
            idxs (list[int]): 标签 ID 列表。

        Returns:
            list[str]: 标签列表。
        """
        return [self.id2tag.get(idx, "O") for idx in idxs]

    def __call__(self, pred):
        """
        计算每种实体类型的精确度、召回率和 F1 分数。

        Args:
            pred: 包含 `label_ids` 和 `predictions` 的预测对象。

        Returns:
            dict[str, float]: 平坦的指标字典，键格式为 '{ENTITY}_{METRIC}'。
        """
        label_ids: torch.Tensor = pred.label_ids
        predictions: torch.Tensor = pred.predictions
        label_ids_batch = torch.tensor(label_ids).cpu().tolist()
        pred_ids_batch = torch.tensor(predictions).argmax(dim=-1).cpu().tolist()
        label_tags_batch = [self.ids2tags(ids) for ids in label_ids_batch]
        pred_tags_batch = [self.ids2tags(ids) for ids in pred_ids_batch]
        label_entities_batch = [set(self.tags2entities(tags)) for tags in label_tags_batch]
        pred_entities_batch = [set(self.tags2entities(tags)) for tags in pred_tags_batch]

        # 初始化计数器
        metrics = {type: {'TP': 0, 'FP': 0, 'FN': 0} for type in self.entity_types}

        for true_set, pred_set in zip(label_entities_batch, pred_entities_batch):
            for type in self.entity_types:
                type_true = {entity for entity in true_set if entity.type == type}
                type_pred = {entity for entity in pred_set if entity.type == type}
                
                metrics[type]['TP'] += len(type_true & type_pred)
                metrics[type]['FP'] += len(type_pred - type_true)
                metrics[type]['FN'] += len(type_true - type_pred)

        results = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for type, counts in metrics.items():
            true_pos = counts['TP']
            false_pos = counts['FP']
            false_neg = counts['FN']

            precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
            recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            results[f'{type}_precision'] = precision
            results[f'{type}_recall'] = recall
            results[f'{type}_f1'] = f1
            
            total_tp += true_pos
            total_fp += false_pos
            total_fn += false_neg
        
        total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if total_precision + total_recall > 0 else 0
        
        results['precision'] = total_precision
        results['recall'] = total_recall
        results['f1'] = total_f1
        
        return results