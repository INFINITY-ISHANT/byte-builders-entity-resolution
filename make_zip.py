"""Build <team_name>_submission.zip in the structure required by the challenge.

    python make_zip.py --team <team_name>

Contents: output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/
(src, README.md, requirements.txt; cache/ and large model binaries excluded) and
Documentation_template.md.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "code" / "business_entity_resolution"


def main():
    """Assemble the submission zip and list its contents."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    a = ap.parse_args()
    out = ROOT / f"{a.team}_submission.zip"
    files = [ROOT / "output" / "matching_results.tsv", ROOT / "output" / "candidate_pairs.tsv",
             ROOT / "Documentation_template.md", PKG / "README.md", PKG / "requirements.txt"]
    files += sorted(p for p in (PKG / "src").glob("*.py"))
    files += sorted(p for p in (PKG / "models").glob("*") if p.suffix in (".json", ".csv", ".txt"))
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"{info.file_size:>14,}  {info.filename}")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
