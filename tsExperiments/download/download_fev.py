#!/usr/bin/env python3
"""
Download fev-bench-mini and GIFT-eval datasets and cache them locally.
The 4 datasets: UCI Air Quality, Hospital Admissions, M-DENSE, Rossmann.
"""

import os
import sys
import time
from pathlib import Path

# Create cache directory
CACHE_DIR = Path(__file__).parent.parent / "fev_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to our fev_cache directory
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / "datasets")

print("="*70)
print("FEV-Bench-Mini and GIFT-eval Dataset Download")
print("="*70)

# Check fev installation
try:
    import fev
    print(f"\n✓ fev library installed (version {fev.__version__})")
except ImportError:
    print("\n✗ fev library NOT installed")
    print("\nInstall with: pip install fev")
    sys.exit(1)

print(f"✓ Cache directory: {CACHE_DIR}")
print(f"✓ HF cache: {CACHE_DIR / 'datasets'}")

def download_dataset(name, short, fev_name, domain, freq, horizon, windows, series, targets, note, target_column=None):
    """Download and cache a single fev dataset."""
    adapted_target_dim = series * targets
    print(f"\n{'='*70}")
    print(f"Dataset {name} ({short})")
    print(f"{'='*70}")
    print(f"  FEV Config: {fev_name}")
    print(f"  Domain: {domain}")
    print(f"  Frequency: {freq}")
    print(f"  Horizon: {horizon}")
    print(f"  Series: {series}, Targets per series: {targets}")
    print(f"  Total target dimensions: {series} × {targets} = {adapted_target_dim}")
    if target_column:
        if isinstance(target_column, list):
            print(f"  Target columns: {', '.join(target_column)}")
        else:
            print(f"  Target column: {target_column}")
    print(f"  Note: {note}")
    
    try:
        print(f"\n  Downloading from HuggingFace Hub...")
        task_kwargs = {
            'dataset_path': "autogluon/fev_datasets",
            'dataset_config': fev_name,
            'horizon': horizon,
            'num_windows': windows,
        }
        if target_column:
            task_kwargs['target'] = target_column
        
        task = fev.Task(**task_kwargs)
        
        # Load full dataset with retry logic for network issues
        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                task.load_full_dataset()
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                # Check if it's a network/timeout error
                if "timeout" in error_str.lower() or "connection" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  ⚠ Network timeout on attempt {attempt + 1}/{max_retries}")
                        print(f"  → Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ✗ Failed after {max_retries} attempts due to network issues")
                        raise
                else:
                    # Not a network error, raise immediately
                    raise
        
        print(f"  ✓ Dataset downloaded and cached")
        print(f"  ✓ Frequency: {task.freq}")
        print(f"  ✓ Horizon: {task.horizon}")
        print(f"  ✓ Num windows: {task.num_windows}")
        
        # Convert first window to GluonTS to verify
        window = task.get_window(0)
        train_ds, test_ds = fev.convert_input_data(window, adapter="gluonts")
        
        train_list = list(train_ds)
        print(f"  ✓ Number of time series in GluonTS: {len(train_list)}")
        
        # Check if we have multivariate data that needs to be split into univariate series
        if len(train_list) > 0:
            first_entry = train_list[0]
            actual_target_dim = first_entry["target"].shape[0] if len(first_entry["target"].shape) > 1 else 1
            print(f"  ✓ Target dim per series: {actual_target_dim}")
            print(f"  ✓ Structure: {len(train_list)} series × {actual_target_dim} targets")
            print(f"  ✓ Start date: {first_entry['start']}")
            
            # If we have a single multivariate series but expect multiple targets, split it
            if len(train_list) == 1 and actual_target_dim > 1 and targets > 1:
                print(f"  → Splitting 1 multivariate series into {actual_target_dim} univariate series")
                # This will be handled during data saving below
        
        # Save processed GluonTS data to disk
        print(f"\n  Saving processed data to disk...")
        import pickle
        import numpy as np
        import pandas as pd
        
        dataset_dir = CACHE_DIR / "processed" / short
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all test windows first (before alignment)
        all_test_data_raw = []
        for window_idx in range(windows):
            window = task.get_window(window_idx)
            _, test_data_window = fev.convert_input_data(window, adapter="gluonts")
            all_test_data_raw.append(list(test_data_window))
        
        # Get train data
        train_data_list = list(train_ds)
        
        # Split multivariate series into separate univariate series if needed
        # (For datasets like UCI Air where multiple targets are in a single series)
        if len(train_data_list) == 1 and train_data_list[0]["target"].ndim == 2 and train_data_list[0]["target"].shape[0] > 1:
            num_dims = train_data_list[0]["target"].shape[0]
            print(f"  → Splitting multivariate data: 1 series with {num_dims} dimensions → {num_dims} univariate series")
            
            # Split train data
            original_entry = train_data_list[0]
            train_data_list = []
            for dim in range(num_dims):
                new_entry = {
                    'start': original_entry['start'],
                    'target': original_entry['target'][dim, :],  # Extract single dimension
                }
                # Copy other fields if they exist
                for key in original_entry.keys():
                    if key not in ['start', 'target']:
                        if isinstance(original_entry[key], np.ndarray) and original_entry[key].ndim > 1:
                            new_entry[key] = original_entry[key][dim, :]
                        else:
                            new_entry[key] = original_entry[key]
                train_data_list.append(new_entry)
            
            # Split test data for all windows
            for window_idx in range(len(all_test_data_raw)):
                window_data = all_test_data_raw[window_idx]
                if len(window_data) == 1 and window_data[0]["target"].ndim == 2 and window_data[0]["target"].shape[0] > 1:
                    original_entry = window_data[0]
                    split_window = []
                    for dim in range(num_dims):
                        new_entry = {
                            'start': original_entry['start'],
                            'target': original_entry['target'][dim, :],
                        }
                        for key in original_entry.keys():
                            if key not in ['start', 'target']:
                                if isinstance(original_entry[key], np.ndarray) and original_entry[key].ndim > 1:
                                    new_entry[key] = original_entry[key][dim, :]
                                else:
                                    new_entry[key] = original_entry[key]
                        split_window.append(new_entry)
                    all_test_data_raw[window_idx] = split_window
        
        # Find global minimum length and latest start date across train and ALL test windows
        all_lengths = []
        all_starts = []
        
        # Get train lengths and start dates
        for entry in train_data_list:
            all_lengths.append(entry["target"].shape[-1])
            all_starts.append(entry["start"])
        
        # Get test lengths and start dates from all windows
        for window_data in all_test_data_raw:
            for entry in window_data:
                all_lengths.append(entry["target"].shape[-1])
                all_starts.append(entry["start"])
        
        if len(all_lengths) > 0:
            min_length = min(all_lengths)
            max_length = max(all_lengths)
            latest_start = max(all_starts)
            earliest_start = min(all_starts)
            
            length_mismatch = min_length != max_length
            start_mismatch = earliest_start != latest_start
            
            if length_mismatch or start_mismatch:
                if length_mismatch:
                    print(f"  ⚠ Length mismatch: min={min_length}, max={max_length}")
                if start_mismatch:
                    print(f"  ⚠ Start date mismatch: earliest={earliest_start}, latest={latest_start}")
                print(f"  → Aligning ALL data to: start={latest_start}, length={min_length}")
                
                # Truncate train data (align both start dates and lengths)
                train_data_aligned = []
                for entry in train_data_list:
                    # Calculate offset from beginning based on start date difference
                    entry_start = pd.Period(entry["start"], freq=task.freq) if hasattr(entry["start"], 'to_timestamp') else pd.Period(entry["start"], freq=task.freq)
                    latest_start_period = pd.Period(latest_start, freq=task.freq) if hasattr(latest_start, 'to_timestamp') else pd.Period(latest_start, freq=task.freq)
                    
                    if entry_start != latest_start_period:
                        # Calculate number of periods between start dates
                        start_offset = (latest_start_period - entry_start).n
                    else:
                        start_offset = 0
                    end_idx = start_offset + min_length
                    
                    # Create a proper deep copy of the entry dict
                    aligned_entry = {}
                    for key, value in entry.items():
                        if key == "target":
                            aligned_entry[key] = np.array(value[..., start_offset:end_idx])
                        elif key in ["feat_dynamic_real", "past_feat_dynamic_real"] and value is not None:
                            aligned_entry[key] = np.array(value[..., start_offset:end_idx])
                        elif key == "start":
                            aligned_entry[key] = latest_start
                        else:
                            aligned_entry[key] = value
                    train_data_aligned.append(aligned_entry)
                train_data_list = train_data_aligned
                
                # Truncate all test windows (align both start dates and lengths)
                all_test_data = []
                for window_idx, window_data in enumerate(all_test_data_raw):
                    window_aligned = []
                    for entry in window_data:
                        # Calculate offset from beginning based on start date difference
                        entry_start = pd.Period(entry["start"], freq=task.freq) if hasattr(entry["start"], 'to_timestamp') else pd.Period(entry["start"], freq=task.freq)
                        latest_start_period = pd.Period(latest_start, freq=task.freq) if hasattr(latest_start, 'to_timestamp') else pd.Period(latest_start, freq=task.freq)
                        
                        if entry_start != latest_start_period:
                            # Calculate number of periods between start dates
                            start_offset = (latest_start_period - entry_start).n
                        else:
                            start_offset = 0
                        end_idx = start_offset + min_length
                        
                        # Create a proper deep copy of the entry dict
                        aligned_entry = {}
                        for key, value in entry.items():
                            if key == "target":
                                aligned_entry[key] = np.array(value[..., start_offset:end_idx])
                            elif key in ["feat_dynamic_real", "past_feat_dynamic_real"] and value is not None:
                                aligned_entry[key] = np.array(value[..., start_offset:end_idx])
                            elif key == "start":
                                aligned_entry[key] = latest_start
                            else:
                                aligned_entry[key] = value
                        window_aligned.append(aligned_entry)
                    all_test_data.append(window_aligned)
            else:
                print(f"  ✓ All data aligned: start={latest_start}, length={min_length}")
                all_test_data = all_test_data_raw
        else:
            all_test_data = all_test_data_raw
        
        # Save train data
        with open(dataset_dir / "train.pkl", "wb") as f:
            pickle.dump(train_data_list, f)
        
        # Save test data
        with open(dataset_dir / "test.pkl", "wb") as f:
            pickle.dump(all_test_data, f)
        
        # Save metadata (use train_data_list after splitting, not original train_list)
        metadata_dict = {
            'freq': task.freq,
            'horizon': task.horizon,
            'num_windows': task.num_windows,
            'num_series': len(train_data_list),
            'target_dim': 1,  # After splitting, each series is univariate (1 target per series)
        }
        with open(dataset_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata_dict, f)
        
        print(f"  ✓ Data saved to: {dataset_dir}")
        print(f"  ✓ Files: train.pkl, test.pkl, metadata.pkl")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

print("\n" + "="*70)
print("Downloading 4 Datasets from fev-bench-mini and GIFT-eval")
print("="*70)

datasets = [
    {
        'name': 'UCI Air Quality',
        'short': 'uci_air',
        'fev_name': 'uci_air_quality_1H',
        'domain': 'nature',
        'freq': 'H (Hourly)',
        'horizon': 24,
        'windows': 20,
        'series': 1,
        'targets': 4,
        'note': 'Hourly data: 24 input/output windows',
        'target_column': ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    },
    {
        'name': 'Hospital Admissions',
        'short': 'hospital',
        'fev_name': 'hospital_admissions_1D',
        'domain': 'healthcare',
        'freq': 'D (Daily)',
        'horizon': 30,
        'windows': 20,
        'series': 8,
        'targets': 1,
        'note': 'Daily data: 30 input/output windows'
    },
    {
        'name': 'M-DENSE',
        'short': 'mdense',
        'fev_name': 'M_DENSE_1D',
        'domain': 'mobility',
        'freq': 'D (Daily)',
        'horizon': 30,
        'windows': 10,
        'series': 30,
        'targets': 1,
        'note': 'Daily data: 30 input/output windows'
    },
    {
        'name': 'Hierarchical Sales',
        'short': 'hierachi',
        'fev_name': 'hierarchical_sales_1D',
        'domain': 'retail',
        'freq': 'D (Daily)',
        'horizon': 30,
        'windows': 10,
        'series': 1115,
        'targets': 1,
        'note': 'Daily data: 30 input/output windows'
    }
]

success_count = 0
for ds in datasets:
    if download_dataset(**ds):
        success_count += 1

print("\n" + "="*70)
print(f"Download Summary: {success_count}/{len(datasets)} datasets ready")
print("="*70)

if success_count == len(datasets):
    print("\n✓ All datasets downloaded and cached successfully!")
    print(f"\nDatasets cached in: {CACHE_DIR}")
    print("\nYou can now train models using:")
    print("  ./train.sh 3141 uci_air timePrism 16 relaxed-wta Full")
    print("  ./train.sh 3141 hospital ETS Short")
    print("  ./train.sh 3141 mdense timeGrad 100 Full")
    print("  ./train.sh 3141 hierachi timeMCL 16 relaxed-wta Short")
else:
    print(f"\n⚠ Warning: Only {success_count} out of {len(datasets)} datasets downloaded successfully")
    print("  Check errors above for details")

if __name__ == "__main__":
    pass

