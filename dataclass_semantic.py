import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class Dataclass(Dataset):
    def __init__(self, simulated_image_dir, real_image_dir, simulated_label_dir):
        # Function to extract sort key from file name (assuming numeric identifiers in filenames)
        def extract_key(filepath):
            # This regex assumes filenames contain numbers that can be used for sorting
            # Adjust the regex pattern if your filenames have a different structure
            match = re.search(r'\d+', os.path.basename(filepath))
            if match:
                return int(match.group())
            else:
                return filepath  # Fallback to using the filepath itself as a key
        
        # List and sort simulated image and label files
        self.simulated_image_paths = sorted(
            [os.path.join(simulated_image_dir, file) for file in os.listdir(simulated_image_dir) if file.endswith('.png')],
            key=extract_key
        )
        self.simulated_label_paths = sorted(
            [os.path.join(simulated_label_dir, file) for file in os.listdir(simulated_label_dir) if file.endswith('.png')],
            key=extract_key
        )

        # List and sort real image and label files
        self.real_image_paths = sorted(
            [os.path.join(real_image_dir, file) for file in os.listdir(real_image_dir) if file.endswith('.jpg')],
            key=extract_key
        )

        # Combine simulated and real paths
        self.image_paths = self.simulated_image_paths + self.real_image_paths

        # Assign domain labels: 0 for simulated, 1 for real
        self.domain_labels = [0] * len(self.simulated_image_paths) + [1] * len(self.real_image_paths)

        assert len(self.image_paths) == len(self.simulated_image_paths) + len(self.real_image_paths), "Mismatch in number of images."
        assert len(self.simulated_image_paths) == len(self.simulated_label_paths), "Mismatch in the number of simulated images and labels."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Handle input images
        image = Image.open(self.image_paths[idx])
        np_image = np.array(image, dtype=np.float32) / 255.0 # Scale pixel values to [0, 1]
        image_tensor = torch.from_numpy(np_image.transpose((2, 0, 1)))  # Convert to [C, H, W] format
        domain_label_tensor = torch.tensor(self.domain_labels[idx], dtype=torch.float32)
        
        #image_file_name = os.path.basename(self.image_paths[idx]) #for debuggin the visualization script

        # Check if the image is from the simulated domain
        if idx < len(self.simulated_image_paths):
            label = Image.open(self.simulated_label_paths[idx])
            np_label = np.array(label, dtype=np.uint8)
            mapped_labels = np.zeros(np_label.shape[:2], dtype=np.int32)  # Initialize all to background class

            # Give the semantic colors their respective classes
            color_to_class_mapping = {
            (0, 255, 0): 0,   # Green
            (255, 0, 239): 1, # Pink
            (0, 0, 255): 2,   # Blue
            (0, 255, 255): 3, # Cyan
            (255, 255, 0): 4, # Yellow
            (255, 0, 0): 5,   # Red
            (0, 0, 0): 6      # Black
            }

            for RGB_value, class_id in color_to_class_mapping.items():
                # Create a mask for pixels equal to the current RGB value
                mask =  (np_label[..., 0] == RGB_value[0]) & \
                        (np_label[..., 1] == RGB_value[1]) & \
                        (np_label[..., 2] == RGB_value[2])
                # Apply mask to assign class ID
                mapped_labels[mask] = class_id  

            label_tensor = torch.from_numpy(mapped_labels).long()
            return image_tensor, label_tensor, domain_label_tensor
        else:
            # For real images, return a dummy label tensor (e.g., all zeros)
            dummy_label_tensor = torch.zeros(image_tensor.shape[1], image_tensor.shape[2], dtype=torch.long)
            return image_tensor, dummy_label_tensor, domain_label_tensor