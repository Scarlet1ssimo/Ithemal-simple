import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('evaluation_results_skl.csv')

# Plot 1: llvm-mca vs measured throughput (Heatmap)
plt.figure(figsize=(10, 8))
plt.hist2d(x=df['measured throughput'], y=df['llvm-mca'], bins=50, cmap='viridis')
plt.colorbar(label='Counts')
plt.title('LLVM-MCA Predicted vs. Actual Throughput (Heatmap)')
plt.xlabel('Measured Throughput (Actual)')
plt.ylabel('LLVM-MCA Predicted Throughput')
# Add a y=x line for reference
min_val = min(df['measured throughput'].min(), df['llvm-mca'].min())
max_val = max(df['measured throughput'].max(), df['llvm-mca'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
plt.savefig('llvm_mca_vs_actual_heatmap.png')
plt.close()

# Plot 2: model vs measured throughput (Heatmap)
plt.figure(figsize=(10, 8))
plt.hist2d(x=df['measured throughput'], y=df['model'], bins=50, cmap='viridis')
plt.colorbar(label='Counts')
plt.title('Model Predicted vs. Actual Throughput (Heatmap)')
plt.xlabel('Measured Throughput (Actual)')
plt.ylabel('Model Predicted Throughput')
# Add a y=x line for reference
min_val_model = min(df['measured throughput'].min(), df['model'].min())
max_val_model = max(df['measured throughput'].max(), df['model'].max())
plt.plot([min_val_model, max_val_model], [min_val_model, max_val_model], 'k--', lw=2)
plt.savefig('model_vs_actual_heatmap.png')
plt.close()

print("Heatmaps saved as llvm_mca_vs_actual_heatmap.png and model_vs_actual_heatmap.png")
