"""
Exp036A — AfriVoices Dataset Profiler

Outputs:
- reports/dataset_analysis/afrivoices_profile.md
- reports/dataset_analysis/afrivoices_language_summary.csv
- reports/dataset_analysis/afrivoices_split_summary.csv
- reports/dataset_analysis/figures/*.png
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files

REPORT_DIR = Path("reports/dataset_analysis")
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ANV_SOURCES = {
    "kik": {"name": "Kikuyu", "repo": "Anv-ke/kikuyu"},
    "luo": {"name": "Luo / Dholuo", "repo": "Anv-ke/Dholuo"},
    "kln": {"name": "Kalenjin", "repo": "Anv-ke/Kalenjin"},
    "mas": {"name": "Maasai", "repo": "Anv-ke/Maasai"},
    "som": {"name": "Somali", "repo": "Anv-ke/Somali"},
}

SWAHILI_REPO = "DigitalUmuganda/Afrivoice_Swahili"


def load_anv_language(iso, info):
    repo = info["repo"]
    files = list_repo_files(repo_id=repo, repo_type="dataset")

    transcript_files = [f for f in files if f.endswith("/files/transcripts.csv")]
    meta_files = [f for f in files if f.endswith("/files/meta.csv")]

    rows = []

    for tx_file in transcript_files:
        parts = tx_file.split("/")
        split = parts[0]
        speech_type = parts[1] if len(parts) > 1 else "unknown"

        tx_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=tx_file)
        tx = pd.read_csv(tx_path)

        matching_meta = tx_file.replace("transcripts.csv", "meta.csv")
        if matching_meta in meta_files:
            meta_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=matching_meta)
            meta = pd.read_csv(meta_path)
            if "recorder_uuid" in tx.columns and "recorder_uuid" in meta.columns:
                tx = tx.merge(meta, on="recorder_uuid", how="left", suffixes=("", "_speaker"))

        tx["iso"] = iso
        tx["language_name"] = info["name"]
        tx["source_repo"] = repo
        tx["split"] = split
        tx["speech_type"] = speech_type
        rows.append(tx)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    df["transcription"] = df.get("actualSentence", "")
    df["translation"] = df.get("translatedText", "")
    df["duration_seconds"] = pd.to_numeric(df.get("duration", None), errors="coerce")
    df["speaker_id"] = df.get("recorder_uuid", "")
    df["dialect"] = df.get("sentenceDialect", df.get("dialect", ""))
    df["gender_final"] = df.get("gender", "")
    df["age_final"] = df.get("ownerAge", "")
    df["audio_ref"] = df.get("mediaPathId", "")

    return df


def load_swahili():
    files = list_repo_files(repo_id=SWAHILI_REPO, repo_type="dataset")
    manifest_files = [f for f in files if f.endswith(".jsonl") and "manifest" in f]

    rows = []

    for mf in manifest_files:
        parts = mf.split("/")
        split_folder = parts[0]

        if "train" in split_folder:
            split = "train"
        elif "dev" in split_folder:
            split = "dev"
        elif "test" in split_folder:
            split = "test"
        else:
            split = "unknown"

        path = hf_hub_download(repo_id=SWAHILI_REPO, repo_type="dataset", filename=mf)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["iso"] = "swa"
                    item["language_name"] = "Swahili"
                    item["source_repo"] = SWAHILI_REPO
                    item["split"] = split
                    item["speech_type"] = "spontaneous"
                    item["manifest_file"] = mf
                    rows.append(item)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    text_candidates = [
        "actualSentence", "transcription", "text", "sentence",
        "transcript", "normalized_text", "raw_transcription"
    ]
    duration_candidates = ["duration", "duration_seconds", "audio_duration"]
    audio_candidates = ["audio_filepath", "audio", "path", "audio_path", "file", "mediaPathId"]

    df["transcription"] = ""
    for col in text_candidates:
        if col in df.columns:
            df["transcription"] = df[col]
            break

    df["duration_seconds"] = pd.NA
    for col in duration_candidates:
        if col in df.columns:
            df["duration_seconds"] = pd.to_numeric(df[col], errors="coerce")
            break

    df["audio_ref"] = ""
    for col in audio_candidates:
        if col in df.columns:
            df["audio_ref"] = df[col]
            break

    df["speaker_id"] = df.get("speaker_id", "")
    df["dialect"] = df.get("dialect", "")
    df["gender_final"] = df.get("gender", "")
    df["age_final"] = df.get("age", "")

    return df


def summarize(df):
    lang_summary = (
        df.groupby(["iso", "language_name"], dropna=False)
        .agg(
            utterances=("transcription", "count"),
            hours=("duration_seconds", lambda x: x.dropna().sum() / 3600),
            mean_duration=("duration_seconds", "mean"),
            median_duration=("duration_seconds", "median"),
            max_duration=("duration_seconds", "max"),
            speakers=("speaker_id", lambda x: x.dropna().nunique()),
        )
        .reset_index()
    )

    split_summary = (
        df.groupby(["iso", "language_name", "split", "speech_type"], dropna=False)
        .agg(
            utterances=("transcription", "count"),
            hours=("duration_seconds", lambda x: x.dropna().sum() / 3600),
        )
        .reset_index()
    )

    lang_summary.to_csv(REPORT_DIR / "afrivoices_language_summary.csv", index=False)
    split_summary.to_csv(REPORT_DIR / "afrivoices_split_summary.csv", index=False)

    return lang_summary, split_summary


def plot_bar(df, x, y, title, filename):
    plt.figure()
    df.plot(kind="bar", x=x, y=y, legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename)
    plt.close()


def make_charts(df, lang_summary, split_summary):
    plot_bar(lang_summary, "iso", "hours", "Hours by Language", "hours_by_language.png")
    plot_bar(lang_summary, "iso", "utterances", "Utterances by Language", "utterances_by_language.png")

    speech = (
        split_summary.groupby("speech_type", dropna=False)["hours"]
        .sum()
        .reset_index()
    )
    plot_bar(speech, "speech_type", "hours", "Hours by Speech Type", "hours_by_speech_type.png")

    duration = df["duration_seconds"].dropna()
    if not duration.empty:
        plt.figure()
        duration.clip(upper=60).hist(bins=50)
        plt.title("Utterance Duration Histogram (clipped at 60s)")
        plt.xlabel("Seconds")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "duration_histogram.png")
        plt.close()

    dialects = (
        df.groupby(["iso", "dialect"], dropna=False)
        .size()
        .reset_index(name="utterances")
        .sort_values("utterances", ascending=False)
        .head(25)
    )
    dialects.to_csv(REPORT_DIR / "afrivoices_top_dialects.csv", index=False)


def write_report(df, lang_summary, split_summary):
    report = REPORT_DIR / "afrivoices_profile.md"

    lines = [
        "# AfriVoices Dataset Profile",
        "",
        "Experiment: Exp036A",
        "",
        "## Overall Summary",
        "",
        f"- Total utterances: {len(df):,}",
        f"- Total measured hours: {df['duration_seconds'].dropna().sum() / 3600:.2f}",
        f"- Languages: {df['iso'].nunique()}",
        f"- Speakers: {df['speaker_id'].dropna().nunique()}",
        "",
        "## Language Summary",
        "",
        lang_summary.to_markdown(index=False),
        "",
        "## Split / Speech-Type Summary",
        "",
        split_summary.to_markdown(index=False),
        "",
        "## Generated Charts",
        "",
        "- `figures/hours_by_language.png`",
        "- `figures/utterances_by_language.png`",
        "- `figures/hours_by_speech_type.png`",
        "- `figures/duration_histogram.png`",
        "",
    ]

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


def main():
    frames = []

    print("Loading Swahili...")
    frames.append(load_swahili())

    for iso, info in ANV_SOURCES.items():
        print(f"Loading {iso}...")
        frames.append(load_anv_language(iso, info))

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    df.to_parquet(REPORT_DIR / "afrivoices_profile_raw.parquet", index=False)

    lang_summary, split_summary = summarize(df)
    make_charts(df, lang_summary, split_summary)
    write_report(df, lang_summary, split_summary)

    print("Exp036A profiling complete.")


if __name__ == "__main__":
    main()
