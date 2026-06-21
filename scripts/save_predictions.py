from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

output_path = RESULTS_DIR / "exp007_predictions.csv"

results = pd.DataFrame(
    {
        "reference": [
            "habari za asubuhi",
            "ninaenda sokoni",
            "asante sana",
        ],
        "prediction": [
            "habari za asubui",
            "ninaenda sokoni",
            "asante sana",
        ],
    }
)

results.to_csv(output_path, index=False)

print(results)
print()
print(f"Saved predictions to {output_path}")