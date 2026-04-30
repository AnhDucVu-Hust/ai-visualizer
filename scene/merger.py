"""
Segment merger: collapses consecutive transcription segments into groups
bounded by min/max duration thresholds.

Merging rules
-------------
1. Accumulate segments into the current group.
2. When the group's total duration >= ``min_duration``, finalize it and
   start a new group for the next segment.
3. If adding the *next* segment would push the group over ``max_duration``,
   finalize the current group first — even if it hasn't reached min_duration.
4. Whatever remains in the last open group is always emitted.

This means every output group satisfies:
  duration >= min_duration   OR   adding one more segment would exceed max_duration
"""

from __future__ import annotations

import logging
from typing import List

from stt.models import Segment, TranscriptionResult
from .models import MergedSegment, SceneConfig

logger = logging.getLogger(__name__)


def merge_segments(
    result: TranscriptionResult,
    config: SceneConfig,
) -> List[MergedSegment]:
    """
    Merge transcription segments from *result* according to *config* thresholds.

    Parameters
    ----------
    result:
        Output of any ``BaseSTTEngine.transcribe()`` call.
    config:
        Scene config carrying ``min_duration`` and ``max_duration``.

    Returns
    -------
    List[MergedSegment]
        Ordered list of merged groups ready for prompt generation.
    """
    segments: List[Segment] = result.segments
    if not segments:
        logger.warning("merge_segments: TranscriptionResult contains no segments.")
        return []

    min_dur = config.min_duration
    max_dur = config.max_duration

    if min_dur > max_dur:
        raise ValueError(
            f"min_duration ({min_dur}s) must be <= max_duration ({max_dur}s)"
        )

    groups: List[MergedSegment] = []

    # State for the current open group
    group_start: float = segments[0].start
    group_end: float = segments[0].end
    group_texts: List[str] = [segments[0].text]
    group_ids: List[int] = [segments[0].id]

    def _flush(gid: int) -> MergedSegment:
        return MergedSegment(
            group_id=gid,
            start=group_start,
            end=group_end,
            text=" ".join(t.strip() for t in group_texts),
            source_segment_ids=list(group_ids),
        )

    for seg in segments[1:]:
        prospective_end = seg.end
        prospective_duration = prospective_end - group_start
        current_duration = group_end - group_start

        # Would adding this segment breach the hard ceiling?
        would_exceed_max = prospective_duration > max_dur

        if would_exceed_max:
            # Finalize the current group before accepting the new segment.
            groups.append(_flush(len(groups)))
            logger.debug(
                "Flushed group %d  [%.2fs -> %.2fs]  dur=%.2fs  (max exceeded)",
                len(groups) - 1,
                group_start,
                group_end,
                current_duration,
            )
            # Start a fresh group with the current segment.
            group_start = seg.start
            group_end = seg.end
            group_texts = [seg.text]
            group_ids = [seg.id]

        elif current_duration >= min_dur:
            # We've already satisfied the minimum — finalize and start fresh.
            groups.append(_flush(len(groups)))
            logger.debug(
                "Flushed group %d  [%.2fs -> %.2fs]  dur=%.2fs  (min reached)",
                len(groups) - 1,
                group_start,
                group_end,
                current_duration,
            )
            group_start = seg.start
            group_end = seg.end
            group_texts = [seg.text]
            group_ids = [seg.id]

        else:
            # Still below min — keep accumulating.
            group_end = seg.end
            group_texts.append(seg.text)
            group_ids.append(seg.id)

    # Emit whatever remains in the last open group.
    if group_texts:
        groups.append(_flush(len(groups)))
        logger.debug(
            "Flushed final group %d  [%.2fs -> %.2fs]  dur=%.2fs",
            len(groups) - 1,
            group_start,
            group_end,
            group_end - group_start,
        )

    logger.info(
        "merge_segments: %d segments → %d groups  "
        "(min=%.1fs  max=%.1fs)",
        len(segments),
        len(groups),
        min_dur,
        max_dur,
    )
    return groups
