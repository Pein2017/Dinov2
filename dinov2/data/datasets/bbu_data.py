import json
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image, ImageFile

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class BBUDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        split: str,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the BBUDataset.

        Args:
            split (str): One of 'train', 'val', or 'test' indicating the dataset split.
            root (str): Root directory of the dataset.
            transforms (Optional[Callable]): Optional transform to be applied on a sample.
            transform (Optional[Callable]): Optional transform to be applied on the image.
            target_transform (Optional[Callable]): Optional transform to be applied on the target.
        """
        super().__init__(root, transforms, transform, target_transform)
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.split = split
        self.annotations_path = os.path.join(root, 'annotations', f'{split}.json')
        self._load_annotations()

    def _load_annotations(self):
        """
        Loads the annotation JSON file and stores the data list.
        """
        with open(self.annotations_path, 'r') as f:
            data = json.load(f)
        self.data_list = data['data_list']

    def get_image_data(self, index: int) -> bytes:
        """
        Retrieves the raw image data for a given index.

        Args:
            index (int): Index of the sample.

        Returns:
            bytes: Raw image data.
        """
        img_path = os.path.join(self.root, self.data_list[index]['img_path'])
        with open(img_path, 'rb') as f:
            return f.read()

    def get_target(self, index: int) -> Optional[int]:
        """
        Retrieves the target label for a given index.

        Args:
            index (int): Index of the sample.

        Returns:
            Optional[int]: Ground truth label (0 for negative, 1 for positive).
        """
        return self.data_list[index]['gt_label']

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Retrieves the image and target for a given index.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Any, Any]: Tuple containing the transformed image and its target.
        """
        image_data = self.get_image_data(index)

        try:
            image = ImageDataDecoder(image_data).decode()
        except OSError as e:
            # Handle the error, e.g., skip the image or log the error
            print(f"Error loading image {self.data_list[index]['img_path']}: {e}")
            # Optionally, you can skip the corrupted image by recursively calling __getitem__
            return self.__getitem__((index + 1) % len(self))

        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
