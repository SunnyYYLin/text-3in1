from configs import ConfigParser
from models import get_model
from datasets import get_datasets, get_collators
from metrics import get_metrics
from transformers.trainer import Trainer

if __name__ == '__main__':
    # reading config
    parser = ConfigParser()
    config = parser.parse_config()
    print(config)
    
    # prepare for training
    ## data
    train_dataset, val_dataset, test_dataset = get_datasets(config)
    print(f"train_dataset: {len(train_dataset)}, \
        \b\bval_dataset: {len(val_dataset)}, \
        \b\btest_dataset: {len(test_dataset)}")
    collator = get_collators(config)
    ## model
    model = get_model(config)
    ## trainer
    train_args = config.train_args()
    metrics = get_metrics(config)
    trainer = Trainer(model=model, 
                      args=train_args, 
                      train_dataset=train_dataset, 
                      eval_dataset=val_dataset, 
                      data_collator=collator,
                      compute_metrics=metrics)
    
    if config.mode == 'train':
        # training
        trainer.train()
    
    # testing
    results = trainer.evaluate(test_dataset)
    print(results)