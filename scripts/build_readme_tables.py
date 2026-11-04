"""Rewrite every `<!-- table:NAME -->` block of README.md and docs/*.md from results/tables/ and every `<!-- caption:NAME -->` block from results/figures/figure_manifest.csv (see readme_tables.py)."""
from __future__ import annotations
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_tables as rt

def main() -> None:
    texts = {f: f.read_text() for f in rt.DOCS if f.exists()}
    for kind, names, fill in (("table", list(rt.TABLES), lambda n: rt.render(*rt.TABLES[n]())), ("caption", rt.CAPTIONS, rt.figure_provenance)):
        for name in names:
            block = re.compile(rf"(<!-- {kind}:{name} -->\n)(?:.*?\n)?(<!-- /{kind}:{name} -->)", re.S)
            hit = False
            for f, text in texts.items():
                if block.search(text):
                    hit = True
                    texts[f] = block.sub(lambda m: m.group(1) + fill(name) + "\n" + m.group(2), text)

            if not hit:
                print(f"warning: no marker for {kind} {name!r} in any of {[f.name for f in texts]}")

    for f, text in texts.items():
        f.write_text(text)

    print(f"rebuilt {len(rt.TABLES)} tables and {len(rt.CAPTIONS)} figure captions in {[f.name for f in texts]}")


if __name__ == "__main__":
    main()
