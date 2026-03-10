import os
import json
import glob

# Path to experiment folders
base_dir = 'experiments'

# Find all results.json files (one per experiment)
json_files = glob.glob(os.path.join(base_dir, '**', 'results.json'), recursive=True)

experiment_scores = []

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        experiment = data.get('experiment', file)
        avg_acc = data.get('accuracy', None)
        if avg_acc is not None:
            experiment_scores.append({'experiment': experiment, 'mean_accuracy': avg_acc})

# Sort by mean accuracy descending
experiment_scores.sort(key=lambda x: x['mean_accuracy'], reverse=True)

# Print results
print(f"{'Experiment':<60} {'Mean Accuracy':>15}")
for entry in experiment_scores:
    print(f"{entry['experiment']:<60} {entry['mean_accuracy']:>15.4f}")
