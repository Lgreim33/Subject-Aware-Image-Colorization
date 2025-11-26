import os
import glob
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms

import colorization
import error_map_utils


# Default directories
_BASE_DIR = os.path.dirname(__file__)
grayscale_dir = os.path.join(_BASE_DIR, 'images_512/')
color_dir = os.path.join(_BASE_DIR, 'images_512_gray/')
attention_map_dir = os.path.join(_BASE_DIR, 'attention_maps/')

batch_size = 16


class ColorAttentionDataset(Dataset):
    """Dataset that returns (grayscale_input, color_target, attention_map).

    Assumes files are matched by filename across the three directories.
    """

    def __init__(self, grayscale_dir, color_dir, attention_dir, transform=None):
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.attention_dir = attention_dir
        self.transform = transform

        # Debug: show which directories are being used and whether they exist
        print(f'[dataset] grayscale_dir="{self.grayscale_dir}" exists={os.path.isdir(self.grayscale_dir)}')
        print(f'[dataset] color_dir="{self.color_dir}" exists={os.path.isdir(self.color_dir)}')
        print(f'[dataset] attention_dir="{self.attention_dir}" exists={os.path.isdir(self.attention_dir)}')

        # Collect filenames from each directory. ASSUMPTION:
        # corresponding datapoints are in the same order across all three folders.
        # Find all files recursively (no extension filtering).
        def list_files_sorted(d):
            # Recursively find all files (no extension filtering)
            files = glob.glob(os.path.join(d, '**', '*'), recursive=True)
            # Filter to only regular files (not directories)
            files = [f for f in files if os.path.isfile(f)]
            files = sorted(files)
            return files

        g_list = list_files_sorted(self.grayscale_dir)
        c_list = list_files_sorted(self.color_dir)
        a_list = list_files_sorted(self.attention_dir)

        print(f'Found {len(g_list)} grayscale, {len(c_list)} color, and {len(a_list)} attention map files.')
        if g_list:
            print(f'  [sample grayscale] {g_list[0]}')
        if c_list:
            print(f'  [sample color] {c_list[0]}')
        if a_list:
            print(f'  [sample attention] {a_list[0]}')
        min_len = min(len(g_list), len(c_list), len(a_list))
        if min_len == 0:
            raise RuntimeError('No files found in one of the provided directories')

        # If lengths differ, truncate to the shortest so indices remain aligned
        if not (len(g_list) == len(c_list) == len(a_list)):
            g_list = g_list[:min_len]
            c_list = c_list[:min_len]
            a_list = a_list[:min_len]

        self.samples = list(zip(g_list, c_list, a_list))

        # Default transforms if none provided
        if self.transform is None:
            # Grayscale -> 1xHxW, Color -> 3xHxW, Attention -> 1xHxW
            self.to_tensor = transforms.ToTensor()
        else:
            self.to_tensor = self.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g_path, c_path, a_path = self.samples[idx]

        g_img = Image.open(g_path).convert('L')     # grayscale
        c_img = Image.open(c_path).convert('RGB')   # color (3 channels)
        # Attention map may be stored as an image or a NumPy (.npy) file.
        if a_path.lower().endswith('.npy'):
            a_np = np.load(a_path)
            # Ensure float32 and 2D (H, W)
            a_np = a_np.astype('float32')
            if a_np.ndim == 3 and a_np.shape[2] == 1:
                a_np = a_np[:, :, 0]
            a_t = torch.from_numpy(a_np)
            # Add channel dimension -> (1, H, W)
            if a_t.ndim == 2:
                a_t = a_t.unsqueeze(0)
        else:
            a_img = Image.open(a_path).convert('L')     # attention map (single channel)
            a_t = self.to_tensor(a_img)

        g_t = self.to_tensor(g_img)
        c_t = self.to_tensor(c_img)

        return g_t, c_t, a_t


def get_dataset(grayscale_dir=grayscale_dir, color_dir=color_dir, attention_map_dir=attention_map_dir,
                split=0.8, transform=None, seed=42):
    """Return (train_dataset, test_dataset).

    - grayscale_dir, color_dir, attention_map_dir: directories containing matching files.
    - split: fraction used for training (e.g., 0.8)
    """

    dataset = ColorAttentionDataset(grayscale_dir, color_dir, attention_map_dir, transform=transform)

    total = len(dataset)
    if total == 0:
        raise RuntimeError('Dataset is empty')

    if not (0.0 < split < 1.0):
        raise ValueError('split must be between 0 and 1')

    train_len = int(total * split)
    test_len = total - train_len

    # deterministic split using manual seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=generator)

    return train_dataset, test_dataset


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)




def attention_masked_mae(outputs, targets, attention_map, threshold=0.1, eps=1e-6, soft=False):
    """Compute MAE between `outputs` and `targets`, but only over high-attention pixels.

    - outputs: Tensor[B, C, H, W]
    - targets: Tensor[B, C, H, W]
    - attention_map: Tensor[B, 1, H, W] or [B, H, W]
    - threshold: float in [0,1]. Pixels with normalized attention <= threshold are ignored.
    - soft: if True, use attention as soft weight instead of hard mask (no threshold).

    Returns a scalar tensor (mean loss over selected pixels). If no pixels pass the
    threshold, falls back to global MAE to avoid NaNs.
    """

    # Ensure attention_map tensor shape is (B, 1, H, W)
    if attention_map.ndim == 3:
        attention = attention_map.unsqueeze(1)
    else:
        attention = attention_map

    attention = attention.float()

    # Normalize per-sample to [0,1]
    B = attention.shape[0]
    att_norm = torch.zeros_like(attention)
    for i in range(B):
        a = attention[i]
        a_min = torch.min(a)
        a_max = torch.max(a)
        denom = (a_max - a_min) if (a_max - a_min) > 0 else 1.0
        att_norm[i] = (a - a_min) / (denom + eps)

    if soft:
        weights = att_norm
    else:
        weights = (att_norm > threshold).float()

    # Compute per-pixel MAE across color channels -> shape (B, 1, H, W)
    per_pixel_error = torch.mean(torch.abs(outputs - targets), dim=1, keepdim=True)

    masked_error = per_pixel_error * weights

    selected = torch.sum(weights)
    if selected.item() > 0:
        loss = torch.sum(masked_error) / (selected + eps)
    else:
        loss = torch.mean(per_pixel_error)

    return loss


def inverse_attention_masked_mae(outputs, grayscale_inputs, attention_map, threshold=0.1, eps=1e-6, soft=False):
    """Compute MAE between `outputs` and the grayscale input (expanded to 3 channels),
    but only over regions indicated by the INVERTED attention map.

    This encourages the model to keep non-attended regions close to the original
    grayscale (desaturated) appearance.

    - outputs: Tensor[B, C, H, W]
    - grayscale_inputs: Tensor[B, 1, H, W] or [B, H, W]
    - attention_map: Tensor[B, 1, H, W] or [B, H, W]
    - threshold / soft: same semantics as in attention_masked_mae
    """

    # Normalize and invert the provided attention map using utilities
    att_norm = error_map_utils.normalize_error_map(attention_map.float(), epsilon=eps)
    inv = error_map_utils.invert_error_map(att_norm)

    # Ensure shape (B,1,H,W)
    if inv.ndim == 3:
        inv = inv.unsqueeze(1)

    if grayscale_inputs.ndim == 3:
        g = grayscale_inputs.unsqueeze(1)
    else:
        g = grayscale_inputs

    # Expand grayscale to 3 channels to compare with outputs
    if g.shape[1] == 1 and outputs.shape[1] == 3:
        g_color = g.repeat(1, 3, 1, 1)
    else:
        # If channel counts already match, use directly
        g_color = g

    # Create weights/mask from inverted attention
    if soft:
        weights = inv.float()
    else:
        weights = (inv > threshold).float()

    per_pixel_error = torch.mean(torch.abs(outputs - g_color), dim=1, keepdim=True)
    masked_error = per_pixel_error * weights

    selected = torch.sum(weights)
    if selected.item() > 0:
        loss = torch.sum(masked_error) / (selected + eps)
    else:
        loss = torch.mean(per_pixel_error)

    return loss


def train_colorization_model(model, dataloader, num_epochs, criterion, optimizer, device,
                            att_loss_weight=1.0, inv_loss_weight=1.0, threshold=0.1, soft=False):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f'Epoch: {epoch + 1}')
        for i, data in enumerate(dataloader):
            # Get inputs and move to device
            inputs, targets, error_maps = data
            inputs = inputs.to(device)          # Grayscale images
            targets = targets.to(device)        # Color images (ground truth)
            error_maps = error_maps.to(device)  # Error maps

            # Zero the parameter gradients
            optimizer.zero_grad()
            print(f'  Batch {i+1}/{len(dataloader)}')
            # Forward pass
            outputs = model(inputs)
            print('Output shape:', outputs.shape)

            # Compute attention-masked MAE loss (for attended regions)
            loss_att = attention_masked_mae(outputs, targets, error_maps, threshold=threshold, soft=soft)

            # Compute inverse-attention-masked MAE loss (for non-attended regions)
            loss_inv = inverse_attention_masked_mae(outputs, inputs, error_maps, threshold=threshold, soft=soft)

            # Combine losses (weights can be tuned)
            loss = att_loss_weight * loss_att + inv_loss_weight * loss_inv

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(dataloader):.4f}')



def main():
    # Example usage
    train_dataset, test_dataset = get_dataset()
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    model = colorization.UNet_Colorizer(n_channels_in=1, n_channels_out=3)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start training (training uses attention-masked MAE by default)
    train_colorization_model(model, train_loader, num_epochs=10, criterion=criterion, optimizer=optimizer, device=device)


if __name__ == '__main__':
    main()