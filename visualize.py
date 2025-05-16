import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # Import colors for LogNorm

# Load the data
df = pd.read_csv('evaluation_results_skl.csv')

# Filter the DataFrame
df = df[df['measured throughput'] < 1000]

# Plot 1: llvm-mca vs measured throughput (Heatmap)
plt.figure(figsize=(6,6))  # Adjust figure size for a more square aspect ratio
# sns.histplot(x=df['measured throughput'], y=df['llvm-mca'], bins=50, cmap='viridis', cbar=True)
plt.hist2d(x=df['measured throughput'], y=df['llvm-mca'], bins=50, cmap='Reds', norm=colors.LogNorm(), range=[[0, 1000], [0, 1000]]) # Add range
plt.colorbar(label='Counts')
plt.title('LLVM-MCA vs. Measured Throughput (SKL)')  # Updated title
plt.xlabel('Measured Throughput')
plt.ylabel('LLVM-MCA Predicted Throughput')
# Add a y=x line for reference
min_val = min(df['measured throughput'].min(), df['llvm-mca'].min())
max_val = max(df['measured throughput'].max(), df['llvm-mca'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
plt.xlim(0, 1000)  # Set x-axis limit based on filtering
plt.ylim(0, 1000)  # Set y-axis limit for consistency with x-axis and example
plt.gca().set_aspect('equal', adjustable='box')  # Ensure square plot
plt.savefig('llvm_mca_vs_actual_heatmap.png')
plt.savefig('llvm_mca_vs_actual_heatmap.eps')
plt.close()

# Plot 2: model vs measured throughput (Heatmap)
plt.figure(figsize=(6,6))  # Adjust figure size
# sns.histplot(x=df['measured throughput'], y=df['model'], bins=50, cmap='viridis', cbar=True)
plt.hist2d(x=df['measured throughput'], y=df['model'], bins=50, cmap='Reds', norm=colors.LogNorm(), range=[[0, 1000], [0, 1000]]) # Add range
plt.colorbar(label='Counts')
plt.title('Model vs. Measured Throughput (SKL)')  # Updated title
plt.xlabel('Measured Throughput')
plt.ylabel('Model Predicted Throughput')
# Add a y=x line for reference
min_val_model = min(df['measured throughput'].min(), df['model'].min())
max_val_model = max(df['measured throughput'].max(), df['model'].max())
plt.plot([min_val_model, max_val_model], [
         min_val_model, max_val_model], 'k--', lw=2)
plt.xlim(0, 1000)  # Set x-axis limit
plt.ylim(0, 1000)  # Set y-axis limit
plt.gca().set_aspect('equal', adjustable='box')  # Ensure square plot
plt.savefig('model_vs_actual_heatmap.png')
plt.savefig('model_vs_actual_heatmap.eps')
plt.close()
