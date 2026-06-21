from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

output_path = RESULTS_DIR / "exp007_predictions.csv"

results = pd.DataFrame(
    columns=[
        "reference",
        "prediction",
    ]
)

results.to_csv(output_path, index=False)

print(f"Prediction file created: {output_path}")