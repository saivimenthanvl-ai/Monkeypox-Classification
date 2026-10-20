"""Read-only image dataset audit: counts, exact duplicate hashes and filenames.

This does not prove patient/specimen independence: images can be near-identical
despite different SHA-256 hashes. Pass the dataset root with one folder per class.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def inspect(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    counts, groups, errors = Counter(), defaultdict(list), []
    for folder in sorted(x for x in root.iterdir() if x.is_dir()):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1048576), b""):
                        h.update(chunk)
                groups[h.hexdigest()].append(str(path.relative_to(root)))
                counts[folder.name] += 1
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
    duplicate_groups = [v for v in groups.values() if len(v) > 1]
    return {"root": str(root), "images": sum(counts.values()), "class_counts": dict(counts),
            "exact_duplicate_groups": duplicate_groups,
            "duplicate_groups_spanning_classes": [g for g in duplicate_groups if len({x.split("/")[0] for x in g}) > 1],
            "read_errors": errors, "caveat": "Hash audit cannot detect perceptual duplicates or shared patients/specimens."}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = inspect(args.dataset_root)
    body = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body + "\n", encoding="utf-8")
    print(body)
    if report["read_errors"] or report["duplicate_groups_spanning_classes"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
