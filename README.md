# sample_entropy

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import antropy as ant
import datetime

# === Paths ===
voxelwise_dir = Path('/home/stubanadean/voxelwise_timeseries_nilearn')  # voxelwise .npy files
output_dir = Path('/home/stubanadean/sample_entropy_results')
output_dir.mkdir(exist_ok=True)

# === Layers ===
layers = ['Interoception', 'Exteroception', 'Cognition']

# === Sample Entropy parameters ===
m = 2  # embedding dimension
metric = 'chebyshev'  # distance metric
r_factor = 0.2  # radius factor (r = r_factor * std)

# === Storage ===
results_list = []

# === Loop over layers and ROI files ===
for layer in layers:
    layer_dir = voxelwise_dir / layer
    roi_files = list(layer_dir.glob("*_voxelwise_timeseries.npy"))
    
    if not roi_files:
        print(f"No voxelwise time series found for layer {layer}, skipping...")
        continue
    
    for ts_file in roi_files:
        roi_name = ts_file.stem.replace(f"{ts_file.stem.split('_')[-1]}", "")  # get ROI name cleanly
        ts_data = np.load(ts_file)  # shape: time x voxels
        
        # Compute sample entropy for each voxel
        samp_ent_voxels = []
        for v in range(ts_data.shape[1]):
            cycle_diff = ts_data[:, v]  # time series for one voxel
            r = r_factor * np.std(cycle_diff)
            try:
                se = ant.sample_entropy(cycle_diff, order=m, metric=metric)
                samp_ent_voxels.append(se)
            except Exception as e:
                print(f"  Warning: Could not compute sample entropy for voxel {v} in {roi_name}: {e}")
        
        samp_ent_voxels = np.array(samp_ent_voxels)
        samp_ent_voxels = samp_ent_voxels[np.isfinite(samp_ent_voxels)]  # remove NaNs or infs
        
        if len(samp_ent_voxels) == 0:
            print(f"  Warning: No valid sample entropy values for ROI {roi_name}, skipping.")
            continue
        
        # Store summary stats for this ROI
        results_list.append({
            'Layer': layer,
            'ROI': roi_name,
            'Mean_SampleEntropy': np.mean(samp_ent_voxels),
            'Median_SampleEntropy': np.median(samp_ent_voxels),
            'Std_SampleEntropy': np.std(samp_ent_voxels),
            'Min_SampleEntropy': np.min(samp_ent_voxels),
            'Max_SampleEntropy': np.max(samp_ent_voxels),
            'Num_voxels': len(samp_ent_voxels)
        })

# === Save CSV summary ===
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
csv_path = output_dir / f'sample_entropy_summary_{timestamp}.csv'
results_df = pd.DataFrame(results_list)
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Sample entropy summary saved to {csv_path.resolve()}")
