# AfriVoices Dataset Profile

Experiment: Exp036A

Status: STRUCTURE DEFINED — statistics pending Colab profiling

## Competition Languages

| ISO | Language | Family | Source Count | Sources |
|-----|----------|--------|--------------|---------|
| swa | Swahili | Bantu | 1 | `DigitalUmuganda/Afrivoice_Swahili` |
| kik | Kikuyu | Bantu | 1 | `Anv-ke/kikuyu` |
| luo | Luo / Dholuo | Nilotic Western | 1 | `Anv-ke/Dholuo` |
| som | Somali | Cushitic | 2 | `Anv-ke/Somali`, `DigitalUmuganda/Afrivoice` |
| mas | Maasai | Nilotic Eastern | 1 | `Anv-ke/Maasai` |
| kln | Kalenjin | Nilotic Southern | 1 | `Anv-ke/Kalenjin` |

## Important Design Decision

The profiler and manifest builder are organized around the six official Kaggle evaluation languages, not merely the raw source repositories.

Somali is treated as one competition language (`som`) with two source repositories:

- `Anv-ke/Somali` for Maxatire
- `DigitalUmuganda/Afrivoice` for Mogadishu

This preserves Kaggle scoring alignment while still retaining source and dialect metadata.

## Pending Statistics

- Utterance count per language
- Total hours per language
- Scripted vs spontaneous distribution
- Dialect distribution
- Speaker count
- Gender distribution
- Duration statistics
- Missing-value report
