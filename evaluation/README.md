# Metrics Consolidator

## Quick Start

To consolidate all metric results into a single CSV file:

```bash
cd /path/to/LOGen
python3 evaluation/metrics_consolidator.py
```

The results will be written to `evaluation/metrics_consolidated.csv`

## Overview

The `metrics_consolidator.py` script consolidates metric results from the `evaluation/evaluation/` directory structure into a single CSV file matching the format of `metrics_consolidator_target.csv`.

## Expected Directory Structure

Metrics are expected to be organized as:

```
evaluation/
└── evaluation/
    ├── 1NN_COV/
    │   ├── <experiment_name>/
    │   │   ├── <model_name>/
    │   │   │   └── epoch_<N>/
    │   │   │       └── *.txt
    ├── CD_EMD/
    │   ├── <experiment_name>/
    │   │   ├── <model_name>/
    │   │   │   └── epoch_<N>/
    │   │   │       └── *.txt
    ├── class_acc/
    ├── fid/
    ├── jsd/
    └── kid/
```

## Supported Metrics

The script parses the following metric types:

### 1NN_COV (1-Nearest Neighbor Coverage)
- **Columns**: `gan_mmd-EMD`, `lgan_cov-EMD`, `lgan_mmd_smp-EMD`, `1-NN-EMD-acc_t`, `1-NN-EMD-acc_f`, `1-NN-EMD-acc`
- **Format**: Each metric on a separate line with format `metric_name: value`

### CD_EMD (Chamfer Distance & Earth Mover Distance)
- **Columns**: `CD_mean`, `CD_std`, `EMD_mean`, `EMD_std`
- **Format**: 
  ```
  CD mean <<< 0.1171816951 >>> and std <<< 0.0537617076 >>> 
  EMD mean <<< 0.1002397463 >>> and std <<< 0.0667757094 >>>
  ```

### class_acc (Classification Accuracy)
- **Columns**: `cls_acc_gt` (ground truth), `cls_acc_gen` (generated)
- **Files**: Two files per epoch (`real=True.txt` and `real=False.txt`)
- **Format**:
  ```
  Correct instance number: 2861
  Total instance number: 8006
  Accuracy is  0.35735698226330254
  ```

### FID (Fréchet Inception Distance)
- **Columns**: `FID`
- **Format**: `FID <<< 208.1398010254 >>>`

### KID (Kernel Inception Distance)
- **Columns**: `KID mean`, `KID std`
- **Format**: `KID mean <<< 0.3929715157 >>> and std <<< 0.0429834984 >>>`

### JSD (Jensen-Shannon Divergence)
- **Columns**: `JSD mean`, `JSD std`
- **Format**: `JSD mean <<< 0.1356700212 >>> and std <<< 0.0095902756 >>>`

## Output CSV Format

The output CSV has the following columns:
- `experiment_name`: Experiment name
- `model_name`: Model name
- `epoch`: Training epoch
- All metric columns (empty if not available)
- `details`: Optional details column

Example:
```csv
experiment_name,model_name,epoch,gan_mmd-EMD,lgan_cov-EMD,...,FID,KID mean,KID std,JSD mean,JSD std,details
LOGen_human_experiments_mgpus,gen_sloper4d_map,24,,,,,,0.1171816951,0.0537617076,0.9997501874,0.3574818886,0.1002397463,0.0667757094,206.8545227,,,,,
```

## Script Details

### Main Functions

- `consolidate_metrics(output_csv)`: Main function that orchestrates the consolidation
- `find_metric_files()`: Discovers all metric result files in the evaluation directory
- `parse_metric_file()`: Routes to appropriate parser based on metric type
- `parse_<metric_type>()`: Individual parsers for each metric format

### How It Works

1. Scans `evaluation/evaluation/` for all metric directories
2. For each metric type, finds all `.txt` files in the structure `<metric>/<exp_name>/<model_name>/epoch_<N>/`
3. Parses each file according to its format
4. Groups metrics by (experiment_name, model_name, epoch)
5. Writes consolidated results to CSV with all columns, leaving empty cells for missing metrics

### Extending

To add support for a new metric:

1. Create a parser function: `def parse_<metric_type>(content: str) -> Dict[str, Any]`
2. The parser should extract metrics and return a dictionary with metric names as keys
3. Ensure the metric_type directory exists in `evaluation/evaluation/`
4. Add the expected column names to the `columns` list in `consolidate_metrics()`

## Notes

- The script handles the class_acc metric specially since it produces two files per run (real=True and real=False)
- Only real=False files are processed to avoid duplicates
- Missing metrics are left as empty cells in the CSV
- Metrics are sorted by experiment_name, model_name, and epoch in the output
