"""In-memory preprocessing helpers for the PJM isolated-gloss dataset."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EOR_TOKEN = "EoR"
BLANK_GLOSS = "blank"


def default_gloss_map_path():
    """Return the default gloss map path bundled with the annotations."""
    return Path(__file__).resolve().parent / "annotations" / "gloss_map.json"


def load_gloss_map(gloss_map_path=None):
    """Load optional gloss canonicalization map without modifying dataset files."""
    gloss_map_path = Path(gloss_map_path) if gloss_map_path is not None else default_gloss_map_path()
    if not gloss_map_path.exists():
        return {}

    with open(gloss_map_path, encoding="utf-8") as f:
        return json.load(f)


def map_gloss(gloss_name, gloss_map):
    """Return canonical gloss name using gloss_map, falling back to the original name."""
    return gloss_map.get(gloss_name, gloss_name)


class VideoCache:
    """Lazy in-memory cache for raw .npy videos."""

    def __init__(self, data_dir, enabled=True):
        self.data_dir = Path(data_dir)
        self.enabled = enabled
        self.cache = {}

    def load(self, stem):
        """Load a raw video by stem and keep it in RAM when caching is enabled."""
        if self.enabled and stem in self.cache:
            return self.cache[stem]

        npy_path = self.data_dir / f"{stem}.npy"
        if not npy_path.exists():
            return None

        raw = np.load(npy_path, allow_pickle=True)
        if self.enabled:
            self.cache[stem] = raw
        return raw


def load_split_segments(split_file):
    """Read split file into {stem: [(start, gloss), ...]}."""
    stem_segments = defaultdict(list)
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            stem, start_str, gloss = parts[0], parts[1], parts[2]
            stem = stem.replace(".npy", "")
            stem_segments[stem].append((int(start_str), gloss))

    return stem_segments


def load_timestamped_segments(annotation_dir, stem):
    """Return list of (gloss, start) and the EoR frame for one annotation file."""
    json_path = Path(annotation_dir) / f"{stem}.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("glosses", [])
    if not entries or not isinstance(entries[0], list):
        return [], None

    normalized_entries = [(gloss, int(start)) for gloss, start in entries]
    eor = None
    for gloss, start in normalized_entries:
        if gloss == EOR_TOKEN:
            eor = start
            break

    return normalized_entries, eor


def build_samples(
    data_dir,
    annotation_dir,
    split_file,
    gloss2id,
    gloss_map_path=None,
    add_blank_segments=True,
    cache_videos=True,
):
    """Build dataset samples in memory and return them with a shared video cache."""
    gloss_map = load_gloss_map(gloss_map_path)
    video_cache = VideoCache(data_dir, enabled=cache_videos)
    samples = []
    stem_segments = load_split_segments(split_file)

    for stem, segs in stem_segments.items():
        entries, eor = load_timestamped_segments(annotation_dir, stem)
        if not entries:
            continue

        regular_entries = [(g, s) for g, s in entries if g != EOR_TOKEN]

        if add_blank_segments:
            blank_gid = gloss2id.get(BLANK_GLOSS)
            if blank_gid is not None:
                if regular_entries:
                    first_start = min(start for _, start in regular_entries)
                    if first_start > 0:
                        samples.append((stem, 0, first_start, blank_gid))
                elif eor is not None and eor > 0:
                    samples.append((stem, 0, eor, blank_gid))

                raw = video_cache.load(stem)
                video_len = len(raw) if raw is not None else None
                if eor is not None and video_len is not None and video_len > eor:
                    samples.append((stem, eor, video_len, blank_gid))

        seg_map = {}
        for i, (gloss, start) in enumerate(entries):
            if gloss == EOR_TOKEN:
                continue
            next_start = entries[i + 1][1] if i + 1 < len(entries) else eor
            seg_map[(gloss, start)] = next_start

        for start, gloss_name in segs:
            end = seg_map.get((gloss_name, start))
            if end is not None and end > start:
                mapped_gloss = map_gloss(gloss_name, gloss_map)
                gid = gloss2id.get(mapped_gloss)
                if gid is not None:
                    samples.append((stem, start, end, gid))

    return samples, video_cache
