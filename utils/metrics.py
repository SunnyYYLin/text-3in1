import torch
from torchmetrics import Accuracy, F1Score
from configs import BaseConfig

def get_metrics(config: BaseConfig):
    """
    根据任务类型返回适当的 compute_metrics 函数
    """
    # 情感分类（例如，二分类）
    if config.task == "sentiment":
        accuracy_metric = Accuracy(task="binary" 
            if config.num_classes == 2 else "multiclass", 
            num_classes=config.num_classes).to(config.device)
        f1_metric = F1Score(task="binary" 
            if config.num_classes == 2 else "multiclass", 
            num_classes=config.num_classes).to(config.device)
        
        def compute_metrics(pred):
            labels = torch.tensor(pred.label_ids, device=config.device)
            preds = torch.tensor(pred.predictions, device=config.device).argmax(dim=-1)

            accuracy = accuracy_metric(preds, labels)
            f1 = f1_metric(preds, labels)
            return {
                'accuracy': accuracy.item(),
                'f1': f1.item(),
            }
    
    # 命名实体识别（NER，多标签分类）
    elif config.task == "ner":
        accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.num_classes, average='macro').to(config.device)
        f1_metric = F1Score(task="multiclass", 
            num_classes=config.num_classes, average='macro').to(config.device)

        def compute_metrics(pred):
            labels = torch.tensor(pred.label_ids).to(config.device)
            preds = torch.tensor(pred.predictions).argmax(dim=-1).to(config.device)

            accuracy = accuracy_metric(preds, labels)
            f1 = f1_metric(preds, labels)
            return {
                'accuracy': accuracy.item(),
                'f1': f1.item(),
            }

    # 下一个词预测任务（next-token，通常是语言模型）
    elif config.task == "next-token":
        accuracy_metric = Accuracy(task="multiclass", 
            num_classes=config.vocab_size).to(config.device)

        def compute_metrics(pred):
            labels = torch.tensor(pred.label_ids).to(config.device)
            preds = torch.tensor(pred.predictions).argmax(dim=-1).to(config.device)

            accuracy = accuracy_metric(preds, labels)
            return {
                'accuracy': accuracy.item()
            }
    
    else:
        raise ValueError(f"Unsupported task type: {config.task}")

    return compute_metrics

def extract_BIO(seq):
    """
    Extracts the BIO tag from a sequence.
    Args:
        seq: List of BIO tags, e.g., ['B-PER', 'I-PER', 'O', 'B-LOC']
    Returns:
        res: A list of tuples (tag, chunk_start, chunk_end)
    """
    res = []
    chunk_start = None
    current_tag = None

    for i, tag in enumerate(seq):
        if tag.startswith("B-"):  # Beginning of a chunk
            if current_tag is not None:
                # Store the previous chunk
                res.append((current_tag, chunk_start, i - 1))
            current_tag = tag[2:]  # Extract tag type (e.g., 'PER', 'LOC')
            chunk_start = i
        elif tag.startswith("I-") and current_tag == tag[2:]:
            # Inside a chunk, continue
            continue
        else:  # End of a chunk or outside
            if current_tag is not None:
                res.append((current_tag, chunk_start, i - 1))
                current_tag = None
            chunk_start = None
    
    # If the last tag forms a chunk, append it
    if current_tag is not None:
        res.append((current_tag, chunk_start, len(seq) - 1))

    return res