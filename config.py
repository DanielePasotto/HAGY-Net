import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Unified Configuration for HAGY-Net and HarmNet.")
    
    # General Execution
    parser.add_argument("--harmnet", type=bool, default=False, help="Train/Test HarmNet")
    parser.add_argument("--hagynet", type=bool, default=True, help="Train/Test HAGY-Net")
    parser.add_argument("--both", type=bool, default=False, help="Train/Test both HarmNet and HAGY-Net")
    parser.add_argument("--fine_tune", type=bool, default=False, help="Fine tune HAGY-Net")
    
    # Paths & Directories
    parser.add_argument("--hagynet_weights_path", type=str, default="weights/hagynet_weights.pth", help="Path to HAGY-Net weights")
    parser.add_argument("--harmnet_weights_path", type=str, default="weights/harmnet_weights.pth", help="Path to HarmNet weights")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path where is located dataset")
    parser.add_argument("--processed_dataset_path", type=str, default="processed_dataset", help="Path where is located processed dataset")
    
    # Model Architecture
    parser.add_argument("--input_shape", type=int, default=1, help="Input shape")
    parser.add_argument("--hidden_units", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--output_shape", type=int, default=1, help="Output shape")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Losses
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha for Tversky loss")
    parser.add_argument("--beta", type=float, default=0.2, help="Beta for Tversky loss")
    parser.add_argument("--lambda_mae", type=float, default=1.0, help="Lambda for MAE loss")
    parser.add_argument("--lambda_ssim", type=float, default=0.5, help="Lambda for SSIM loss")
    parser.add_argument("--lambda_tv", type=float, default=0.1, help="Lambda for TV loss")
    parser.add_argument("--lambda_edge", type=float, default=0.1, help="Lambda for edge loss")
    parser.add_argument("--lambda_bg", type=float, default=0.5, help="Lambda for background loss")
    
    # Callbacks
    parser.add_argument("--scheduler_mode", type=str, default="max", help="Scheduler mode")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--early_stopping_mode", type=str, default="max", help="Early stopping mode")
    
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args
