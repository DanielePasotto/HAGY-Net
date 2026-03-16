import os
import torch
import numpy as np
import dataset
from timeit import default_timer as timer
from tqdm import tqdm
from utils.metrics import SemanticEvaluator
from utils.loss_functions import TverskyBCELoss, RegionAwareCompositeLoss
from networks.segmentation_model import HAGYNet, HarmNet
from torch import optim
from utils.utils import EarlyStopping, training, validation
from torch.utils.data import DataLoader
from config import get_args

def train_hagynet(args, train_loader, val_loader, device):
    model = HAGYNet(args.input_shape, args.hidden_units, args.output_shape, device, args.harmnet_weights_path)
    if args.fine_tune:
        model.load_state_dict(torch.load(args.hagynet_weights_path, map_location=device, weights_only=True))
        print("### Model loaded for fine-tuning ###")
    evaluator = SemanticEvaluator(num_classes=args.num_classes, device=device)
    criterion = TverskyBCELoss(args.alpha, args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, args.scheduler_mode, patience=args.scheduler_patience)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode=args.early_stopping_mode)
    
    best_val_dice = float('-inf')

    print(f"Starting training on {device}...")

    start_time = timer()
    for epoch in range(args.epochs):
        train_loss = training(model, train_loader, criterion, optimizer, device)
        val_metrics = validation(model, val_loader, criterion, device, evaluator)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Global Accuracy: {val_metrics['Global Accuracy']:.2f} | "
            f"mIoU: {val_metrics['mIoU']:.3f} | "
            f"mDICE: {val_metrics['mDICE']:.3f} | "
            f"Precision: {val_metrics['Precision']:.3f} | "
            f"Recall: {val_metrics['Recall']:.3f}")

        if val_metrics['mDICE'] > best_val_dice:
            best_val_dice = val_metrics['mDICE']
            torch.save(model.state_dict(), os.path.join(args.weights_dir, 'hagynet_weights.pth'))
            print("--> Model saved (Best Val mDICE)")
        
        scheduler.step(val_metrics['mDICE'])
        early_stopping(val_metrics['mDICE'])

        if early_stopping.early_stop:
            print("\n" + "#"*30)
            print(f"EARLY STOPPING TRIGGERED at Epoch {epoch+1}")
            print(f"The model did not improve mDICE for {early_stopping.patience} epochs.")
            print("#"*30)
            break 

    end_time = timer()
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

def train_harmnet(args, train_loader, val_loader, device):
    model = HarmNet(input_shape=args.input_shape, hidden_units=args.hidden_units, output_shape=args.output_shape).to(device)
    criterion = RegionAwareCompositeLoss(lambda_mae=args.lambda_mae, lambda_ssim=args.lambda_ssim, lambda_tv=args.lambda_tv, lambda_edge=args.lambda_edge, lambda_bg=args.lambda_bg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, args.scheduler_mode, patience=args.scheduler_patience)
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode=args.early_stopping_mode)

    best_val_ssim = float('-inf')

    print(f"Starting training on {device}...")

    start_time = timer()
    for epoch in range(args.epochs):
        train_loss = training(model, train_loader, criterion, optimizer, device)
        val_metrics = validation(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} | " 
            f"SSIM: {val_metrics['ssim']:.3f} | " 
            f"CNR: {val_metrics['cnr']:.2f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"BG Roughness: {val_metrics['bg_roughness']:.4f}") 

        if val_metrics['ssim'] > best_val_ssim:
            best_val_ssim = val_metrics['ssim']
            torch.save(model.state_dict(), os.path.join(args.weights_dir, 'harmnet_weights.pth'))
            print("--> Model saved (Best Val SSIM)")
        
        scheduler.step(val_metrics['ssim'])

        early_stopping(val_metrics['ssim'])
        
        if early_stopping.early_stop:
            print("\n" + "!"*30)
            print(f"EARLY STOPPING TRIGGERED at Epoch {epoch+1}")
            print(f"The model did not improve SSIM for {early_stopping.patience} epochs.")
            print("!"*30)
            break 

    end_time = timer()
    total_time = end_time - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs(args.weights_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.harmnet or args.both:
        train_dataset = dataset.OrganoidDataset(os.path.join(args.processed_dataset_path, "train"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, target=True)
        val_dataset = dataset.OrganoidDataset(os.path.join(args.processed_dataset_path, "val"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, target=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        train_harmnet(args, train_loader, val_loader, device)
    if args.hagynet or args.both:
        train_dataset = dataset.OrganoidDataset(os.path.join(args.dataset_path, "train"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, train=True)
        val_dataset = dataset.OrganoidDataset(os.path.join(args.dataset_path, "val"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, train=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        train_hagynet(args, train_loader, val_loader, device)

if __name__ == "__main__":
    main()