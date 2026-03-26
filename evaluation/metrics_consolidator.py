#!/usr/bin/env python3
"""
Consolidate metric results from evaluation/evaluation/<metric>/<experiment_name>/<model_name>/<epoch_id>/<metric>.txt
into a single CSV file matching the format in metrics_consolidator_target.csv
"""

import os
import csv
import glob
import re
from collections import defaultdict
from typing import Dict, Tuple, Any


def parse_1nn_cov(content: str) -> Dict[str, Any]:
    """Parse 1NN coverage metric file"""
    metrics = {}
    lines = content.strip().split('\n')
    for line in lines:
        if ':' in line:
            match = re.search(r'([a-zA-Z0-9\-_]+):\s*([\d.]+)', line)
            if match:
                metric_name = match.group(1).strip()
                value = float(match.group(2))
                metrics[metric_name] = value
    return metrics


def parse_cd_emd(content: str) -> Dict[str, Any]:
    """Parse CD_EMD metric file"""
    metrics = {}
    cd_match = re.search(r'CD mean\s*<<<\s*([\d.]+)\s*>>>\s*and std\s*<<<\s*([\d.]+)\s*>>>', content)
    if cd_match:
        metrics['CD_mean'] = float(cd_match.group(1))
        metrics['CD_std'] = float(cd_match.group(2))
    
    emd_match = re.search(r'EMD mean\s*<<<\s*([\d.]+)\s*>>>\s*and std\s*<<<\s*([\d.]+)\s*>>>', content)
    if emd_match:
        metrics['EMD_mean'] = float(emd_match.group(1))
        metrics['EMD_std'] = float(emd_match.group(2))
    return metrics


def parse_class_acc(filepath: str) -> Dict[str, Any]:
    """Parse classification accuracy files (reads both real=True and real=False)"""
    metrics = {}
    base_path = filepath.replace('_real=False.txt', '').replace('_real=True.txt', '')
    
    # Read real=True file (ground truth)
    real_true_path = base_path + '_real=True.txt'
    if os.path.exists(real_true_path):
        try:
            with open(real_true_path, 'r') as f:
                content = f.read()
                match = re.search(r'Accuracy is\s+([\d.]+)', content)
                if match:
                    metrics['cls_acc_gt'] = float(match.group(1))
        except Exception as e:
            print(f"Warning: Error reading {real_true_path}: {e}")
    
    # Read real=False file (generated)
    real_false_path = base_path + '_real=False.txt'
    if os.path.exists(real_false_path):
        try:
            with open(real_false_path, 'r') as f:
                content = f.read()
                match = re.search(r'Accuracy is\s+([\d.]+)', content)
                if match:
                    metrics['cls_acc_gen'] = float(match.group(1))
        except Exception as e:
            print(f"Warning: Error reading {real_false_path}: {e}")
    return metrics


def parse_fid(content: str) -> Dict[str, Any]:
    """Parse FID metric file"""
    metrics = {}
    fid_match = re.search(r'FID\s*<<<\s*([\d.]+)\s*>>>', content)
    if fid_match:
        metrics['FID'] = float(fid_match.group(1))
    return metrics


def _parse_kid(content: str) -> Dict[str, Any]:
    """Parse KID metric file"""
    metrics = {}
    kid_mean_match = re.search(r'KID mean:\s*<<<\s*([\d.]+)\s*>>>', content)
    if kid_mean_match:
        metrics['KID mean'] = float(kid_mean_match.group(1))
    
    kid_std_match = re.search(r'KID std\s*<<<\s*([\d.]+)\s*>>>', content)
    if kid_std_match:
        metrics['KID std'] = float(kid_std_match.group(1))
    return metrics

def parse_kid(content: str) -> Dict[str, Any]:
    """Parse KID metric file"""
    metrics = {}
    jsd_match = re.search(r'KID mean\s*<<<\s*([\d.]+)\s*>>>\s*and std\s*<<<\s*([\d.]+)\s*>>>', content)
    if jsd_match:
        metrics['KID mean'] = float(jsd_match.group(1))
        metrics['KID std'] = float(jsd_match.group(2))
    return metrics


def parse_jsd(content: str) -> Dict[str, Any]:
    """Parse JSD metric file"""
    metrics = {}
    jsd_match = re.search(r'JSD mean\s*<<<\s*([\d.]+)\s*>>>\s*and std\s*<<<\s*([\d.]+)\s*>>>', content)
    if jsd_match:
        metrics['JSD mean'] = float(jsd_match.group(1))
        metrics['JSD std'] = float(jsd_match.group(2))
    return metrics


def find_metric_files(base_dir: str = './evaluation/evaluation'):
    """Find all metric result files"""
    metric_files = {}
    metric_types = ['1NN_COV', 'CD_EMD', 'class_acc', 'fid', 'jsd', 'kid']
    
    for metric_type in metric_types:
        metric_base = os.path.join(base_dir, metric_type)
        if not os.path.exists(metric_base):
            continue
        
        pattern = os.path.join(metric_base, '*', '*', '*', '*.txt')
        for filepath in glob.glob(pattern):
            parts = filepath.split(os.sep)
            try:
                metric_idx = parts.index(metric_type)
                experiment_name = parts[metric_idx + 1]
                model_name = parts[metric_idx + 2]
                epoch_str = parts[metric_idx + 3]
                
                # Parse epoch (format: "epoch_XX")
                epoch_match = re.search(r'epoch_(\d+)', epoch_str)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                else:
                    try:
                        epoch = int(epoch_str)
                    except ValueError:
                        continue
                
                # Skip duplicate class_acc entries (only keep real=False)
                if metric_type == 'class_acc' and 'real=True' in filepath:
                    continue
                
                key = (experiment_name, model_name, epoch, metric_type)
                metric_files[key] = filepath
            except (ValueError, IndexError):
                continue
    
    return metric_files


def parse_metric_file(filepath: str, metric_type: str) -> Dict[str, Any]:
    """Parse a metric file"""
    if metric_type == 'class_acc':
        return parse_class_acc(filepath)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return {}
    
    if metric_type == '1NN_COV':
        return parse_1nn_cov(content)
    elif metric_type == 'CD_EMD':
        return parse_cd_emd(content)
    elif metric_type == 'fid':
        return parse_fid(content)
    elif metric_type == 'kid':
        return parse_kid(content)
    elif metric_type == 'jsd':
        return parse_jsd(content)
    
    return {}


def consolidate_metrics(output_csv: str = './evaluation/metrics_consolidated.csv'):
    """Consolidate all metrics into a CSV file"""
    
    # Find all metric files
    metric_files = find_metric_files()
    print(f"Found {len(metric_files)} metric file(s)")
    
    # Group by (experiment_name, model_name, epoch)
    consolidated = defaultdict(lambda: defaultdict(dict))
    
    for (experiment_name, model_name, epoch, metric_type), filepath in sorted(metric_files.items()):
        metrics = parse_metric_file(filepath, metric_type)
        key = (experiment_name, model_name, epoch)
        consolidated[key].update(metrics)
        print(f"  ✓ {metric_type:12} {experiment_name:40} {model_name:25} epoch {epoch:3} → {len(metrics)} metric(s)")
    
    # Define expected columns
    columns = [
        'experiment_name', 'model_name', 'epoch',
        'lgan_mmd-EMD', 'lgan_cov-EMD', 'lgan_mmd_smp-EMD',
        '1-NN-EMD-acc_t', '1-NN-EMD-acc_f', '1-NN-EMD-acc',
        'CD_mean', 'CD_std', 'cls_acc_gt', 'cls_acc_gen',
        'EMD_mean', 'EMD_std', 'FID', 'KID mean', 'KID std',
        'JSD mean', 'JSD std', 'details'
    ]
    
    # Write CSV
    print(f"\nWriting to {output_csv}")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for experiment_name, model_name, epoch in sorted(consolidated.keys()):
            row = {
                'experiment_name': experiment_name,
                'model_name': model_name,
                'epoch': epoch,
            }
            row.update(consolidated[(experiment_name, model_name, epoch)])
            writer.writerow(row)
    
    print(f"✓ Complete! Wrote {len(consolidated)} row(s)")


if __name__ == '__main__':
    consolidate_metrics()
