"""Check every number quoted in README.md tables, and the listed prose claims, against the source CSVs. Exit status 1 on any mismatch."""
from __future__ import annotations
import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_tables as rt

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?", re.I)


def tolerance(token: str) -> float:
    """Half a unit in the last displayed digit."""
    t = token.replace(",", "")
    mant, _, exp = t.lower().partition("e")
    dec = len(mant.split(".")[1]) if "." in mant else 0

    return 0.5 * 10 ** (-dec) * (10 ** int(exp) if exp else 1.0)


def value(token: str) -> float:
    return float(token.replace(",", ""))


def close(a: str, b: str) -> bool:
    return abs(value(a) - value(b)) <= max(tolerance(a), tolerance(b)) * (1 + 1e-9)


def parse_table(lines: list[str]) -> list[list[str]]:
    return [[c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))] for ln in lines if not re.match(r"^\|\s*:?-{3,}", ln)]


def cell_mismatch(readme_cell: str, expected: str) -> bool:
    if readme_cell == expected:
        return False

    a, b = NUM.findall(readme_cell), NUM.findall(expected)
    if len(a) != len(b) or NUM.sub("#", readme_cell) != NUM.sub("#", expected):
        return True

    return not all(close(x, y) for x, y in zip(a, b))


def audit_tables(text: str) -> list[str]:
    errors, managed = [], set()
    for name, fn in rt.TABLES.items():
        m = re.search(rf"<!-- table:{name} -->\n(.*?)\n<!-- /table:{name} -->", text, re.S)
        if not m:
            errors.append(f"table {name}: marker not found in README.md")
            continue

        managed.add(m.start(1))
        got = parse_table(m.group(1).splitlines())
        header, rows = fn()
        exp = [header] + rows
        if len(got) != len(exp):
            errors.append(f"table {name}: {len(got)} rows in README, {len(exp)} from the CSV")
            continue

        for i, (g, e) in enumerate(zip(got, exp)):
            if len(g) != len(e):
                errors.append(f"table {name} row {i}: {len(g)} cells vs {len(e)}")
                continue

            errors += [f"table {name} row {i} col {j}: README {a!r} vs CSV {b!r}" for j, (a, b) in enumerate(zip(g, e)) if cell_mismatch(a, b)]

    for name in rt.CAPTIONS:
        m = re.search(rf"<!-- caption:{name} -->\n(.*?)\n<!-- /caption:{name} -->", text, re.S)
        if not m:
            errors.append(f"caption {name}: marker not found in README.md")
        elif m.group(1) != rt.figure_provenance(name):
            errors.append(f"caption {name}: README text differs from the figure manifest ({m.group(1)[:80]!r}...)")

    # unmanaged tables with numbers
    lines = text.splitlines()
    pos = 0
    block: list[str] = []
    start = 0
    for ln in lines + [""]:
        if ln.startswith("|"):
            if not block:
                start = pos
            block.append(ln)
        elif block:
            body = "\n".join(block)
            inside = any(s <= start <= s + len(body) + 2 or abs(s - start) < 3 for s in managed)
            cells = [c for row in parse_table(block)[1:] for c in row]
            if not inside and any(re.fullmatch(r"[-+]?\d[\d,]*\.?\d*(e[-+]?\d+)?", c) for c in cells):
                errors.append(f"unmanaged table with numeric cells at character {start}: {block[0][:60]!r}")

            block = []

        pos += len(ln) + 1

    return errors


@dataclass
class Claim:
    label: str
    pattern: str  # regex; every capture group is a number quoted in the README
    source: Callable[[], list[float]]  # the CSV values, already scaled to the units displayed in the README


def audit_claims(text: str) -> tuple[list[str], int]:
    errors = []
    try:
        from audit_claims import CLAIMS  # scripts/audit_claims.py, next to the README text it checks
    except ImportError:
        return ["audit_claims.py not found"], 0

    for c in CLAIMS:
        hits = re.findall(c.pattern, text)
        if len(hits) != 1:
            errors.append(f"claim {c.label!r}: pattern matched {len(hits)} times (needs exactly 1)")
            continue

        shown = [hits[0]] if isinstance(hits[0], str) else list(hits[0])
        truth = [float(x) for x in c.source()]
        if len(shown) != len(truth):
            errors.append(f"claim {c.label!r}: {len(shown)} numbers in README, {len(truth)} from the CSV")
            continue

        for tok, t in zip(shown, truth):
            if abs(value(tok) - t) > tolerance(tok) * (1 + 1e-6):
                errors.append(f"claim {c.label!r}: README {tok} vs CSV {t:.6g}")

    return errors, len(CLAIMS)


def diff_csv(new: Path, old: Path) -> int:
    a, b = pd.read_csv(new), pd.read_csv(old)
    n = 0
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        print(f"shape/columns differ: new {a.shape} {list(a.columns)} vs old {b.shape} {list(b.columns)}")
        return 1

    for i in range(len(a)):
        for c in a.columns:
            x, y = a.at[i, c], b.at[i, c]
            same = (x == y) or (pd.isna(x) and pd.isna(y)) or (isinstance(x, float) and isinstance(y, (int, float)) and math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12))
            if not same:
                print(f"row {i} column {c}: new {x!r} vs old {y!r}")
                n += 1

    print(f"{new.name}: {n} differing cells out of {a.size}")

    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", nargs=2, metavar=("NEW", "OLD"))
    args = ap.parse_args()
    if args.diff:
        sys.exit(1 if diff_csv(Path(args.diff[0]), Path(args.diff[1])) else 0)

    text = "\n".join(f.read_text() for f in rt.DOCS if f.exists())  # README.md and docs/*.md are audited as one text
    errs = audit_tables(text)
    cl_errs, n_claims = audit_claims(text)
    errs += cl_errs
    print(f"audited {len(rt.TABLES)} generated tables, {len(rt.CAPTIONS)} figure provenance lines and {n_claims} prose claims")
    for e in errs:
        print("MISMATCH:", e)

    print("AUDIT FAILED" if errs else "AUDIT PASSED")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()