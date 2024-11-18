from functools import partial

import torch

from dinov2.data.collate import collate_data_and_cast
from dinov2.data.loaders import SamplerType, make_data_loader, make_dataset
from dinov2.data.masking import MaskingGenerator


def test_data_loader():
    # Define the dataset string
    dataset_str = "JOINED:root=/path/to/bbu_shield_cleaned:split=train"

    # Create the dataset
    dataset = make_dataset(
        dataset_str=dataset_str,
        transform=None,  # Apply transformations if needed
        target_transform=lambda _: (),
    )

    # Define masking generator
    mask_generator = MaskingGenerator(
        input_size=(112, 112),  # Example patch size; adjust as per your configuration
        max_num_patches=50,  # Example value; adjust as needed
    )

    # Create data loader
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(0.1, 0.3),  # Example mask ratios; adjust as per your config
        mask_probability=0.5,
        n_tokens=112 * 112,
        mask_generator=mask_generator,
        dtype=torch.float16,
    )

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        sampler_type=SamplerType.SHARDED_INFINITE,
        collate_fn=collate_fn,
    )

    # Fetch a batch
    for batch in data_loader:
        print("Batch keys:", batch.keys())
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        break


if __name__ == "__main__":
    test_data_loader()
