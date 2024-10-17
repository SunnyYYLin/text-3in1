from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model: nn.Module, train_loader: DataLoader, eval_dataset: DataLoader, config) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    global_step = 0
    best_acc = 0
    writer = SummaryWriter(log_dir=config.log_dir)
    
    for epoch in range(config.num_epoch):
        interval_correct = 0
        interval_step = 0
        for data in tqdm(train_loader, 
                         desc=f"Epoch {epoch}",
                         disable=not config.verbose):
            logits = model(data[0])
            loss = criterion(logits, data[1])
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            tags = torch.argmax(logits, dim=-1)
            interval_correct += torch.sum(tags == data[1]).item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1
            interval_step += 1
            
            if (global_step+1) % config.eval_interval == 0:
                acc = interval_correct / (config.batch_size * interval_step)
                interval_correct = 0
                interval_step = 0
                writer.add_scalar('Accuracy/train', acc, global_step)
        
        acc = interval_correct / (config.batch_size * interval_step)
        writer.add_scalar('Accuracy/train', acc, global_step)
                
        loss, acc = evaluate(model, eval_dataset, config)
        writer.add_scalar('Accuracy/eval', acc, global_step)
        writer.add_scalar('Loss/eval', loss, global_step)
        print(f"Epoch {epoch} - Loss: {loss}, Accuracy: {acc}")
        if acc > best_acc:
            best_acc = acc
            print(f"Saving model with best accuracy: {acc}")
            torch.save(model.state_dict(), config.save_path+f'_epoch{epoch}.pth')
    
    writer.close()
    return best_acc

def evaluate(model: nn.Module, eval_loader: DataLoader, config) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_correct = 0
    total_loss = 0
    step = 0
    with torch.no_grad():
        for data in tqdm(eval_loader, 
                         desc="Evaluating",
                         disable=not config.verbose):
            logits = model(data[0])
            total_loss += criterion(logits, data[1]).item()
            tags = torch.argmax(logits, dim=-1)
            total_correct += torch.sum(tags == data[1]).item()
            step += 1
    loss = total_loss / step
    acc = total_correct / (step*eval_loader.batch_size)
    return loss, acc