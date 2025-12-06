"""
Waterbirds Dataset Loader with Group Labels
Adapted from DFR (Kirichenko et al.) and Group DRO repositories.
Keeps things simple: one file, one class, clear interface.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(train=True, augment=True):
    """Standard ImageNet transforms used by DFR/Group-DRO papers."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


# =============================================================================
# DATASET CLASS
# =============================================================================

class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset with group labels for spurious correlation research.
    
    Groups (4 total):
        0: landbird on land (majority)
        1: landbird on water (minority)  
        2: waterbird on land (minority - hardest!)
        3: waterbird on water (majority)
    
    Args:
        root_dir: Path to waterbird_complete95_forest2water2 folder
        split: 'train', 'val', or 'test'
        transform: Optional transform to apply
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter by split (0=train, 1=val, 2=test)
        split_map = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_map[split]]
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Extract arrays for fast access
        self.filenames = self.metadata['img_filename'].values
        self.labels = self.metadata['y'].values  # 0=landbird, 1=waterbird
        self.places = self.metadata['place'].values  # 0=land, 1=water
        
        # Compute group labels: group = 2*y + place
        # This gives us 4 groups: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        self.groups = 2 * self.labels + self.places
        
        # Store group counts for reference
        self.n_groups = 4
        self.group_counts = np.bincount(self.groups, minlength=4)
        
        print(f"Loaded {split} split: {len(self)} samples")
        print(f"  Group counts: {dict(enumerate(self.group_counts))}")
        print(f"  Worst group: {self.group_counts.argmin()} with {self.group_counts.min()} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image, label, group, index
        return {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'group': torch.tensor(self.groups[idx], dtype=torch.long),
            'index': idx
        }


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_waterbirds_loaders(root_dir, batch_size=32, num_workers=2, augment=True):
    """
    Get train/val/test dataloaders for Waterbirds.
    
    Args:
        root_dir: Path to waterbird_complete95_forest2water2 folder
        batch_size: Batch size (32 recommended for Waterbirds)
        num_workers: DataLoader workers
        augment: Whether to use data augmentation for training
        
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')
        transform = get_transforms(train=is_train, augment=augment and is_train)
        
        dataset = WaterbirdsDataset(
            root_dir=root_dir,
            split=split,
            transform=transform
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train  # Drop last incomplete batch during training
        )
    
    return loaders


# =============================================================================
# DOWNLOAD HELPER (uses WILDS if available)
# =============================================================================

def download_waterbirds(root_dir='./data'):
    """
    Download Waterbirds dataset using WILDS package.
    Returns path to the dataset folder.
    """
    try:
        from wilds import get_dataset
        print("Downloading Waterbirds via WILDS...")
        dataset = get_dataset(dataset='waterbirds', download=True, root_dir=root_dir)
        data_path = os.path.join(root_dir, 'waterbirds_v1.0')
        print(f"Dataset downloaded to: {data_path}")
        return data_path
    except ImportError:
        print("WILDS not installed. Install with: pip install wilds")
        print("Or download manually from: https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz")
        return None


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == '__main__':
    # Test the dataloader
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to waterbird_complete95_forest2water2')
    args = parser.parse_args()
    
    print("\nTesting Waterbirds dataloader...")
    loaders = get_waterbirds_loaders(args.data_dir, batch_size=32)
    
    # Check one batch
    batch = next(iter(loaders['train']))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].shape}")
    print(f"  Groups: {batch['group'].shape}")
    
    # Print group distribution in batch
    groups = batch['group'].numpy()
    print(f"\nGroup distribution in batch: {np.bincount(groups, minlength=4)}")
