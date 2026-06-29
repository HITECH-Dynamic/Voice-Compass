"""
Experiment 036A — AfriVoices Dataset Profiler

Purpose:
Profile the six official AfriVoices competition languages before building
the unified training manifest.

Important:
The competition has 6 evaluation languages but 7 source repositories because
Somali has two sources:
- Anv-ke/Somali = Maxatire
- DigitalUmuganda/Afrivoice = Mogadishu
"""

from pathlib import Path

REPORT_DIR = Path("reports/dataset_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "swa": {
        "name": "Swahili",
        "family": "Bantu",
        "sources": [
            {
                "repo_id": "DigitalUmuganda/Afrivoice_Swahili",
                "source_name": "Afrivoice Swahili",
                "status": "accessible_legacy_script",
                "notes": "Uses manifest JSONL files and tar.xz audio shards.",
            }
        ],
    },
    "kik": {
        "name": "Kikuyu",
        "family": "Bantu",
        "sources": [
            {
                "repo_id": "Anv-ke/kikuyu",
                "source_name": "ANV-KE Kikuyu",
                "status": "accessible",
                "notes": "Parquet audio shards with meta.csv and transcripts.csv.",
            }
        ],
    },
    "luo": {
        "name": "Luo / Dholuo",
        "family": "Nilotic Western",
        "sources": [
            {
                "repo_id": "Anv-ke/Dholuo",
                "source_name": "ANV-KE Dholuo",
                "status": "accessible",
                "notes": "Parquet audio shards with meta.csv and transcripts.csv.",
            }
        ],
    },
    "som": {
        "name": "Somali",
        "family": "Cushitic",
        "sources": [
            {
                "repo_id": "Anv-ke/Somali",
                "source_name": "ANV-KE Somali Maxatire",
                "status": "accessible",
                "notes": "Parquet audio shards with meta.csv and transcripts.csv.",
            },
            {
                "repo_id": "DigitalUmuganda/Afrivoice",
                "source_name": "Afrivoice Somali Mogadishu",
                "status": "accessible_file_repo",
                "notes": "Must be filtered under Somali/* path.",
            },
        ],
    },
    "mas": {
        "name": "Maasai",
        "family": "Nilotic Eastern",
        "sources": [
            {
                "repo_id": "Anv-ke/Maasai",
                "source_name": "ANV-KE Maasai",
                "status": "accessible",
                "notes": "Parquet audio shards with meta.csv and transcripts.csv.",
            }
        ],
    },
    "kln": {
        "name": "Kalenjin",
        "family": "Nilotic Southern",
        "sources": [
            {
                "repo_id": "Anv-ke/Kalenjin",
                "source_name": "ANV-KE Kalenjin",
                "status": "accessible",
                "notes": "Parquet audio shards with meta.csv and transcripts.csv.",
            }
        ],
    },
}


def write_profile_stub() -> None:
    report = REPORT_DIR / "afrivoices_profile.md"

    lines = [
        "# AfriVoices Dataset Profile",
        "",
        "Experiment: Exp036A",
        "",
        "Status: STRUCTURE DEFINED — statistics pending Colab profiling",
        "",
        "## Competition Languages",
        "",
        "| ISO | Language | Family | Source Count | Sources |",
        "|-----|----------|--------|--------------|---------|",
    ]

    for iso, info in LANGUAGES.items():
        sources = ", ".join(f"`{s['repo_id']}`" for s in info["sources"])
        lines.append(
            f"| {iso} | {info['name']} | {info['family']} | "
            f"{len(info['sources'])} | {sources} |"
        )

    lines.extend(
        [
            "",
            "## Important Design Decision",
            "",
            "The profiler and manifest builder are organized around the six official Kaggle evaluation languages, not merely the raw source repositories.",
            "",
            "Somali is treated as one competition language (`som`) with two source repositories:",
            "",
            "- `Anv-ke/Somali` for Maxatire",
            "- `DigitalUmuganda/Afrivoice` for Mogadishu",
            "",
            "This preserves Kaggle scoring alignment while still retaining source and dialect metadata.",
            "",
            "## Pending Statistics",
            "",
            "- Utterance count per language",
            "- Total hours per language",
            "- Scripted vs spontaneous distribution",
            "- Dialect distribution",
            "- Speaker count",
            "- Gender distribution",
            "- Duration statistics",
            "- Missing-value report",
            "",
        ]
    )

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


def main() -> None:
    write_profile_stub()


if __name__ == "__main__":
    main()
