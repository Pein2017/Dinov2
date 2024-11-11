# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from io import BytesIO
from typing import Any, Optional

from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_bytes: bytes):
        self.image_bytes = image_bytes

    def decode(self) -> Image.Image:
        return Image.open(BytesIO(self.image_bytes)).convert("RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self.target = target

    def decode(self) -> Optional[int]:
        return self.target
