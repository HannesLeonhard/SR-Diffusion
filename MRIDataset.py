import os
import glob

import torch
import torch.utils.data as data
import nibabel as nib  # For reading NIfTI MRI files
import numpy as np
import torchio as tio


class MRIDataset(data.Dataset):
    """
    A PyTorch Dataset class for loading 3D MRI scans.

    This dataset scans a directory for NIfTI files (.nii, .nii.gz)
    and loads them as 3D volumes.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the MRI scan subfolders.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Recursively find all NIfTI files
        self.file_paths = glob.glob(
            os.path.join(root_dir, "**", "*.nii.gz"), recursive=True
        )

        if not self.file_paths:
            print(f"Warning: No '.nii.gz' files found in {root_dir}.")
            print(
                "You may need to change the file extension to '.nii' or check the directory."
            )
            self.file_paths = glob.glob(
                os.path.join(root_dir, "**", "*.nii"), recursive=True
            )

        print(f"Found {len(self.file_paths)} NIfTI files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        "(B, C, D, H, W)"
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.file_paths[idx]

        try:
            # Load the NIfTI file
            mri_img = nib.load(img_path)

            # Get the image data as a NumPy array
            # This is typically in (H, W, D) format
            mri_data = mri_img.get_fdata(dtype=np.float32)

            # Convert to PyTorch tensor
            tensor_data = torch.from_numpy(mri_data)

            # Add a channel dimension (C, H, W, D)
            # 3D CNNs in PyTorch expect (N, C, D, H, W)
            # Let's permute to (D, H, W) and add a channel
            # to make it (C, D, H, W)
            if len(tensor_data.shape) == 3:
                # Permute if necessary, e.g., (D, H, W)
                # For this example, we just add a channel: (1, H, W, D)
                # A more common format is (C, D, H, W)
                # mri_data is (H, W, D), so we get (1, H, W, D)
                tensor_data = tensor_data.unsqueeze(0)

            sample = {"image": tensor_data, "filename": img_path}

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error loading file {img_path}: {e}")
            return None


class PairedMRIDataset(data.Dataset):
    """
    PyTorch Dataset for BIDS-compliant paired 64mT (low-res) and 3T (high-res) scans.
    """

    def __init__(self, file_pairs, transform=None):
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        metadata_id, lr_path, hr_path = self.file_pairs[idx]

        # Create a torchio.Subject for pair transformation
        subject = tio.Subject(
            low_res=tio.ScalarImage(lr_path),
            high_res=tio.ScalarImage(hr_path),
        )

        if self.transform:
            subject = self.transform(subject)

        # Extract the tensor data (torchio uses (C, D, H, W) where C=1)
        lr_tensor = subject["low_res"][tio.DATA].squeeze(0)
        hr_tensor = subject["high_res"][tio.DATA].squeeze(0)

        return lr_tensor, hr_tensor, metadata_id
