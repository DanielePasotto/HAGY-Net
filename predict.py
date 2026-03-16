import os
import torch
import numpy as np
import dataset
from timeit import default_timer as timer
from networks.segmentation_model import HAGYNet
from torch.utils.data import DataLoader
from config import get_args
from tqdm import tqdm
from PIL import Image

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.dataset_path, "results"), exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = dataset.OrganoidDataset(args.dataset_path, transform=dataset.IMG_TRANSFORM, target_transform=dataset.MASK_TRANSFORM, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HAGYNet(args.input_shape, args.hidden_units, args.output_shape, device, args.harmnet_weights_path)
    model.load_state_dict(torch.load(args.hagynet_weights_path, map_location=device, weights_only=True))
    print("##### Model loaded correctly #####")
    model.eval()
    
    start_time = timer()
    with torch.no_grad():
        for i, (image, _) in enumerate(tqdm(loader, desc="Predicting")):
            image = image.to(device)
            output = model(image)
            output = (torch.sigmoid(output) > 0.5).long()
            
            img_name = dataset.images[i]
            base_name = os.path.splitext(img_name)[0]
            save_path = os.path.join(args.dataset_path, "results", f"{base_name}_output.png")
            
            output_np = output.squeeze().cpu().numpy()
            output_np = (output_np * 255).astype(np.uint8)
            original_image = Image.open(os.path.join(args.dataset_path, "images", img_name))
            output_image = Image.fromarray(output_np).resize(original_image.size, Image.BILINEAR)
            output_image.save(save_path)
            
    end_time = timer()          
            
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()