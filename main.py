from configs import ConfigParser
from models import get_model
from data_utils import get_datasets, get_collators
from metrics import get_metrics
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from utils.examples import get_examples

if __name__ == '__main__':
    # reading config
    parser = ConfigParser()
    config = parser.parse_config()
    
    # prepare for training
    ## data
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    print(f"train_dataset: {len(train_dataset)}, \
        \b\bval_dataset: {len(val_dataset)}, \
        \b\btest_dataset: {len(test_dataset)}")
    collator = get_collators(config)
    ## model
    model = get_model(config)
    print(config)
    ## trainer
    train_args = config.train_args()
    metrics = get_metrics(config) 
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping,
        early_stopping_threshold=0.001 # 指标变化阈值
    )
    trainer = Trainer(model=model, 
                      args=train_args, 
                      train_dataset=train_dataset, 
                      eval_dataset=val_dataset, 
                      data_collator=collator,
                      compute_metrics=metrics,
                      callbacks=[early_stopping]
                      )
    trainer.label_names = config.label_names
    
    match config.mode:
        case 'train':
            trainer.train()
        case 'test':
            results = trainer.evaluate(test_dataset)
            print(results)
        case 'example':
            examples = get_examples(trainer.model, test_dataset, config)
            print(examples)
            # src_words = "I love you"
            # src_ids = {"input_ids": [train_dataset.src_vocab[word] for word in src_words.split()]}
            # input_dict = collator([src_ids])
            # output = model.generate(**input_dict)
        case _:
            raise NotImplementedError(f"Mode {config.mode} not implemented")