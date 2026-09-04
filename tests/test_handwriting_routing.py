import unittest
from unittest.mock import patch

import numpy as np

import config
from main import classify_image
from preprocessing.clip_preprocessing import has_table_structure


class HandwritingRoutingTests(unittest.TestCase):
    def test_ruled_page_is_not_a_printed_table(self):
        """Notebook rules alone must not activate the printed-form override."""
        page = np.full((600, 400, 3), 255, dtype=np.uint8)
        for y in range(50, 600, 80):
            page[y:y + 2, :] = 0

        self.assertFalse(has_table_structure(page))

    @patch("main._classify_tiles")
    @patch("main.has_table_structure", return_value=False)
    @patch("main.compute_ink_density", return_value=0.2)
    def test_ambiguous_global_printed_score_defers_to_handwriting_tiles(self, _ink, _table, classify_tiles):
        """A modest page-level printed lead must not bypass strong tile evidence."""
        # labels: printed, handwritten, non_document
        classify_tiles.side_effect = [
            np.array([[0.48, 0.38, 0.14]]),
            np.array([[0.05, 0.92, 0.03]] * 6),
        ]
        bundle = {"labels": ["printed", "handwritten", "non_document"]}
        page = np.full((600, 400, 3), 100, dtype=np.uint8)

        with patch.object(config, "MIN_NON_BLANK_TILES_FOR_HANDWRITTEN", 4):
            is_handwritten, details = classify_image(page, bundle)

        self.assertTrue(is_handwritten)
        self.assertEqual(details["decision"], "tile_vote")


if __name__ == "__main__":
    unittest.main()
