from __future__ import annotations

import numpy as np

from torvex_extract.ppocrv6_ocr import PPOCRV6SmallOCR


class FakeRecognitionSession:
    def __init__(self):
        self.input_shapes = []

    def run(self, _outputs, feed):
        batch = next(iter(feed.values()))
        self.input_shapes.append(tuple(batch.shape))

        logits = np.zeros((batch.shape[0], 2, 2), dtype=np.float32)
        logits[:, :, 1] = 5.0
        return [logits]


def test_ppocrv6_recognizes_crops_in_batches():
    ocr = object.__new__(PPOCRV6SmallOCR)
    ocr.characters = ["x"]
    ocr.rec_image_height = 48
    ocr.rec_max_width = 3200
    ocr.rec_batch_size = 2
    ocr.rec_input_name = "x"
    ocr.rec_session = FakeRecognitionSession()

    crops = [
        np.full((24, 60, 3), 255, dtype=np.uint8),
        np.full((24, 90, 3), 255, dtype=np.uint8),
        np.full((24, 120, 3), 255, dtype=np.uint8),
    ]

    results = ocr._recognize_crops(crops)

    assert [text for text, _ in results] == ["x", "x", "x"]
    assert all(score > 0.98 for _, score in results)
    assert len(ocr.rec_session.input_shapes) == 2
    assert ocr.rec_session.input_shapes[0][0] == 2
    assert ocr.rec_session.input_shapes[1][0] == 1
