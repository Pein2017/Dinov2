import json
import os
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image, ImageFile

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class JoinedDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        split: str,
        data_dirs_list: List[str],
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the JoinedDataset.

        Args:
            split (str): One of 'train', 'val', or 'test' indicating the dataset split.
            data_dirs_list (List[str]): List of root directories of the datasets.
            transforms (Optional[Callable]): Optional transform to be applied on a sample.
            transform (Optional[Callable]): Optional transform to be applied on the image.
            target_transform (Optional[Callable]): Optional transform to be applied on the target.
        """
        super().__init__(data_dirs_list, transforms, transform, target_transform)
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.split = split
        self.data_dirs_list = data_dirs_list
        self.annotations_paths = [os.path.join(root, 'annotations', f'{split}.json') for root in data_dirs_list]
        self._load_annotations()

    def _load_annotations(self):
        """
        Loads the annotation JSON files from all roots and combines the data lists.
        """
        combined_data_list = []
        for annotations_path in self.annotations_paths:
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            for item in data['data_list']:
                root = os.path.dirname(annotations_path).replace('annotations', '')
                item['full_img_path'] = os.path.join(root, item['img_path'])
                combined_data_list.append(item)
        self.data_list = combined_data_list

    def get_image_data(self, index: int) -> bytes:
        """
        Retrieves the raw image data for a given index.

        Args:
            index (int): Index of the sample.

        Returns:
            bytes: Raw image data.
        """
        img_path = self.data_list[index]['full_img_path']
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
