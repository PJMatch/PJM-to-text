"""
In-memory preprocessing helpers for the PJM isolated-gloss dataset.

This module handles the loading of raw MediaPipe arrays, parsing of JSON annotations, 
and caching of videos to speed up dataloader operations during training.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EOR_TOKEN = "EoR"
BLANK_GLOSS = "blank"


def default_gloss_map_path() -> Path:
    """
    Returns the default gloss map path bundled with the annotations.

    Returns:
        Path: The resolved absolute path to `gloss_map.json`.
    """
    return Path(__file__).resolve().parent / "annotations" / "gloss_map.json"


def load_gloss_map(gloss_map_path: str = None) -> dict:
    """
    Loads an optional gloss canonicalization map without modifying dataset files.

    Args:
        gloss_map_path (str, optional): Path to the gloss map JSON. Defaults to None.

    Returns:
        dict: A dictionary mapping raw gloss names to their canonical forms.
    """
    gloss_map_path = Path(gloss_map_path) if gloss_map_path is not None else default_gloss_map_path()
    if not gloss_map_path.exists():
        return {}

    with open(gloss_map_path, encoding="utf-8") as f:
        return json.load(f)


def map_gloss(gloss_name: str, gloss_map: dict) -> str:
    """
    Returns the canonical gloss name using the gloss map, falling back to the original name.

    Args:
        gloss_name (str): The raw gloss name from annotations.
        gloss_map (dict): The loaded gloss mapping dictionary.

    Returns:
        str: The canonical gloss string.
    """
    return gloss_map.get(gloss_name, gloss_name)


END_OF_VIDEO = -1


class VideoCache:
    """
    In-memory cache for raw .npy videos with optional main-process warmup.
    Used to prevent the dataloader from thrashing the disk.
    """

    def __init__(self, data_dir: str, enabled: bool = True):
        """
        Initializes the VideoCache.

        Args:
            data_dir (str): Path to the directory containing `.npy` sequence files.
            enabled (bool): Whether caching is active. Defaults to True.
        """
        self.data_dir = Path(data_dir)
        self.enabled = enabled
        self.cache = {}
        self.frozen = False

    def load(self, stem: str) -> np.ndarray:
        """
        Loads a raw video by stem. Writes to cache only before freeze().

        Args:
            stem (str): The filename stem of the video (without extension).

        Returns:
            np.ndarray: The loaded numpy array containing the skeleton frames, 
                or None if the file does not exist.
        """
        if stem in self.cache:
            return self.cache[stem]

        npy_path = self.data_dir / f"{stem}.npy"
        if not npy_path.exists():
            return None

        raw = np.load(npy_path, allow_pickle=True)
        if self.enabled and not self.frozen:
            self.cache[stem] = raw
        return raw

    def warmup(self, stems: list) -> None:
        """
        Preloads all stems in the main process, then freezes the cache for forked workers.

        Args:
            stems (list[str]): A list of video stems to load into RAM.

        Returns:
            None
        """
        if not self.enabled:
            return

        stems = sorted(stems)
        total = len(stems)
        for idx, stem in enumerate(stems, start=1):
            if idx == 1 or idx % 500 == 0 or idx == total:
                print(f"  warming cache: {idx}/{total}", flush=True)
            self.load(stem)

        self.frozen = True
        print(
            f"  cache ready: {len(self.cache)} videos (frozen, read-only for workers)",
            flush=True,
        )


def unique_stems(samples: list) -> list:
    """
    Returns a sorted list of unique video stems referenced by dataset samples.

    Args:
        samples (list[tuple]): The generated dataset samples.

    Returns:
        list[str]: A sorted list of unique string stems.
    """
    return sorted({stem for stem, _, _, _ in samples})


def load_split_segments(split_file: str) -> dict:
    """
    Reads a split file into a dictionary format.

    Args:
        split_file (str): Path to the text file containing the dataset split.

    Returns:
        dict[str, list[tuple[int, str]]]: A dictionary mapping video stems to 
            a list of tuples containing (start_frame, gloss_name).
    """
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


def load_timestamped_segments(annotation_dir: str, stem: str) -> tuple:
    """
    Returns a list of timestamped glosses and the End-of-Record (EoR) frame.

    Args:
        annotation_dir (str): Path to the directory with JSON annotations.
        stem (str): The video stem to load.

    Returns:
        tuple[list[tuple[str, int]], int | None]: A tuple containing:
            - A list of tuples containing (gloss_name, start_frame).
            - The integer frame index for the EoR token (or None if not found).
    """
    json_path = Path(annotation_dir) / f"{stem}.json"
    if not json_path.exists():
        return [], None

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
    data_dir: str,
    annotation_dir: str,
    split_file: str,
    gloss2id: dict,
    gloss_map_path: str = None,
    add_blank_segments: bool = True,
    cache_videos: bool = True,
) -> tuple:
    """
    Builds dataset samples in memory and returns them with a shared video cache.

    Iterates through the split definitions and annotations to define the start 
    and end frames for every isolated sign clip.

    Args:
        data_dir (str): Path to the raw sequence data.
        annotation_dir (str): Path to the JSON annotations.
        split_file (str): Path to the split definition file.
        gloss2id (dict): Dictionary mapping gloss names to IDs.
        gloss_map_path (str, optional): Path to canonical mapping rules. Defaults to None.
        add_blank_segments (bool): Whether to inject <blank> sequences. Defaults to True.
        cache_videos (bool): Whether to initialize the video cache. Defaults to True.

    Returns:
        tuple[list[tuple], VideoCache]: A tuple containing:
            - A list of sample tuples: `(stem, start_frame, end_frame, gloss_id)`.
            - The instantiated and optionally populated VideoCache object.
    """
    gloss_map = load_gloss_map(gloss_map_path)
    video_cache = VideoCache(data_dir, enabled=cache_videos)
    samples = []
    stem_segments = load_split_segments(split_file)
    total_stems = len(stem_segments)

    for idx, (stem, segs) in enumerate(stem_segments.items(), start=1):
        if idx == 1 or idx % 500 == 0 or idx == total_stems:
            print(f"  preprocessing stems: {idx}/{total_stems}", flush=True)

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

                if eor is not None:
                    samples.append((stem, eor, END_OF_VIDEO, blank_gid))

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

    print(f"  built {len(samples)} samples (cached videos: {len(video_cache.cache)})", flush=True)
    return samples, video_cache