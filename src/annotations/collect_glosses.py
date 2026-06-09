

import argparse
import json
from pathlib import Path

EOR_TOKEN = "EoR"


def _is_new_format(glosses):

    return (
        isinstance(glosses, list)
        and len(glosses) > 0
        and isinstance(glosses[0], list)
    )


def collect_glosses(folder):
    folder = Path(folder)
    glosses = set()
    stats = {
        "files_total": 0,
        "files_new_format": 0,
        "files_skipped_old": 0,
        "files_skipped_empty": 0,
        "files_failed": 0,
    }

    for json_path in sorted(folder.glob("*.json")):
        stats["files_total"] += 1
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            stats["files_failed"] += 1
            continue

        entries = data.get("glosses")

        if not entries:
            stats["files_skipped_empty"] += 1
            continue

        if not _is_new_format(entries):
            stats["files_skipped_old"] += 1
            continue

        stats["files_new_format"] += 1
        for pair in entries:
            name = pair[0]
            if name == EOR_TOKEN:
                continue
            glosses.add(name)

    return sorted(glosses), stats


def parse_args():
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default="/pjm/baza_wideo",
        help="Folder with annotation JSON files (default: /pjm/baza_wideo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(script_dir / "gloss_list.json"),
        help="Output JSON file (default: ./gloss_list.json)",
    )
    return parser.parse_args()


def main():
    """Run gloss collection and write the result to a JSON file."""
    args = parse_args()

    glosses, stats = collect_glosses(args.folder)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(glosses, f, ensure_ascii=False, indent=2)

    print(f"Scanned folder      : {args.folder}")
    print(f"Files total         : {stats['files_total']}")
    print(f"Files (new format)  : {stats['files_new_format']}")
    print(f"Skipped (old format): {stats['files_skipped_old']}")
    print(f"Skipped (empty)     : {stats['files_skipped_empty']}")
    print(f"Failed to read      : {stats['files_failed']}")
    print(f"Unique glosses      : {len(glosses)}")
    print(f"Written to          : {output_path}")


if __name__ == "__main__":
    main()
