import os
import torch
import numpy as np
import dataset
from timeit import default_timer as timer
from utils.metrics import SemanticEvaluator
from utils.loss_functions import TverskyBCELoss, RegionAwareCompositeLoss
from networks.segmentation_model import HAGYNet, HarmNet
from utils.utils import validation
from torch.utils.data import DataLoader
from config import get_args

def test_hagynet(args, test_loader, device):
    model = HAGYNet(args.input_shape, args.hidden_units, args.output_shape, device, args.harmnet_weights_path)
    model.load_state_dict(torch.load(args.hagynet_weights_path, map_location=device, weights_only=True))
    print("##### Model loaded correctly #####")
    evaluator = SemanticEvaluator(num_classes=args.num_classes, device=device)
    criterion = TverskyBCELoss(args.alpha, args.beta).to(device)

    print(f"Starting testing on {device}...")

    start_time = timer()
    test_metrics = validation(model, test_loader, criterion, device, evaluator)

    print(f"Test Metrics: \n"
        f"Loss: {test_metrics['loss']:.4f} \n"
        f"Global Accuracy: {test_metrics['Global Accuracy']:.2f} \n"
        f"mIoU: {test_metrics['mIoU']:.3f} \n"
        f"mDICE: {test_metrics['mDICE']:.3f} \n"
        f"Precision: {test_metrics['Precision']:.3f} \n"
        f"Recall: {test_metrics['Recall']:.3f} \n"
        f"{30 * '-'} \n"
        f"Class-wise Precision: {test_metrics['Class Precision']} \n"
        f"Class-wise Recall: {test_metrics['Class Recall']}")

    end_time = timer()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds.")

def test_harmnet(args, test_loader, device):
    model = HarmNet(input_shape=args.input_shape, hidden_units=args.hidden_units, output_shape=args.output_shape).to(device)
    model.load_state_dict(torch.load(args.harmnet_weights_path, map_location=device, weights_only=True))
    print("##### Model loaded correctly #####")
    evaluator = SemanticEvaluator(num_classes=args.num_classes, device=device)
    criterion = RegionAwareCompositeLoss(lambda_mae=args.lambda_mae, lambda_ssim=args.lambda_ssim, lambda_tv=args.lambda_tv, lambda_edge=args.lambda_edge, lambda_bg=args.lambda_bg).to(device)

    print(f"Starting testing on {device}...")

    start_time = timer()
    test_metrics = validation(model, test_loader, criterion, device, evaluator)

    print(f"Test Metrics: \n"
        f"Loss: {test_metrics['loss']:.4f} \n"
        f"PSNR: {test_metrics['psnr']:.2f} \n" 
        f"SSIM: {test_metrics['ssim']:.3f} \n" 
        f"CNR: {test_metrics['cnr']:.2f} \n"
        f"Dice: {test_metrics['dice']:.4f} \n"
        f"BG Roughness: {test_metrics['bg_roughness']:.4f}") 

    end_time = timer()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds.")

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.harmnet or args.both:
        test_dataset = dataset.OrganoidDataset(os.path.join(args.processed_dataset_path, "test"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, target=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_harmnet(args, test_loader, device)

    if args.hagynet or args.both:
        test_dataset = dataset.OrganoidDataset(os.path.join(args.dataset_path, "test"), transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, train=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_hagynet(args, test_loader, device)

if __name__ == "__main__":
    main()