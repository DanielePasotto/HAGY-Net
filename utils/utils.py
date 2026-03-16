import torch
import torch.nn.functional as F
from utils.metrics import calculate_harmonization_metrics
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=20, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

    def __call__(self, current_score):
        if self.mode == 'max':
            is_improvement = current_score > (self.best_score + self.min_delta)
        else:
            is_improvement = current_score < (self.best_score - self.min_delta)

        if is_improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def training(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for data in tqdm(loader, desc="Training"):
        if len(data) == 3:
            inputs = data[0].to(device)
            masks = data[1].to(device)
            targets = data[2].to(device)

            outputs = model(inputs)

            loss, _ = criterion(outputs, targets, masks, inputs)
        else:
            inputs = data[0].to(device)
            masks = data[1].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(loader)

def validation(model, loader, criterion, device, evaluator=None):
    model.eval()
    
    with torch.no_grad():
        if evaluator is None:
            metrics = {
                'loss': 0.0,
                'cnr': 0.0,          
                'dice': 0.0,         
                'bg_roughness': 0.0, 
                'ssim': 0.0,         
                'psnr': 0.0          
            }
            for data in tqdm(loader, desc="Validation"):
                inputs = data[0].to(device)
                masks = data[1].to(device)
                targets = data[2].to(device)

                outputs = model(inputs)
        
                loss, ssim_item = criterion(outputs, targets, masks, inputs)
                metrics['loss'] += loss.item()

                batch_metrics = calculate_harmonization_metrics(outputs, masks)
                metrics['cnr'] += batch_metrics['cnr']
                metrics['dice'] += batch_metrics['dice']
                metrics['bg_roughness'] += batch_metrics['bg_roughness']
                metrics['ssim'] += (1.0 - ssim_item) 

                mse = F.mse_loss(outputs * masks, targets * masks)
                psnr = 10 * torch.log10(1 / (mse + 1e-8))
                metrics['psnr'] += psnr.item()
        else:
            evaluator.reset()
            segmentation_loss = 0.0
            for data in tqdm(loader, desc="Validation"):
                inputs = data[0].to(device)
                masks = data[1].to(device)

                outputs = model(inputs)

                loss = criterion(outputs, masks)
                segmentation_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                evaluator.update(preds.squeeze(1), masks.long().squeeze(1))
    
    # Average metrics over batches
    num_batches = len(loader)
    if evaluator is None:
        final_metrics = {k: v / num_batches for k, v in metrics.items()}
    else:
        final_metrics = {
            'loss': segmentation_loss / num_batches,
            **evaluator.compute()
        }
    
    return final_metrics
