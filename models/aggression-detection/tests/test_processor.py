from __future__ import annotations

import numpy as np

from processor import interpolate_anchor_scores


def test_anchor_scores_change_on_every_intermediate_frame() -> None:
    scores = interpolate_anchor_scores([0, 4, 8], [0.2, 0.8, 0.4], 9)
    assert np.allclose(
        scores,
        [0.2, 0.35, 0.5, 0.65, 0.8, 0.7, 0.6, 0.5, 0.4],
    )
    assert len(np.unique(scores)) > 3
