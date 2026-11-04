"""Check every relative markdown link and image in README.md, CONTRIBUTING.md and docs/*.md: the file must exist and any #anchor must match a heading."""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def anchors(path: Path) -> set[str]:
    out = set()
    for ln in path.read_text().splitlines():
        if ln.startswith("#"):
            h = ln.lstrip("#").strip().lower()
            out.add(re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", h)))

    return out


def main() -> None:
    bad = []
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"] + sorted((ROOT / "docs").glob("*.md"))
    for f in files:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", f.read_text()):
            if re.match(r"[a-z]+://", target):
                continue

            path, _, frag = target.partition("#")
            dest = (f.parent / path).resolve() if path else f
            if not dest.exists():
                bad.append(f"{f.relative_to(ROOT)}: missing {target}")
            elif frag and dest.suffix == ".md" and frag not in anchors(dest):
                bad.append(f"{f.relative_to(ROOT)}: no heading for #{frag} in {dest.name}")

    print(f"checked links in {len(files)} files")
    for b in bad:
        print("BROKEN:", b)

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
