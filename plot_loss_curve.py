import re
import matplotlib.pyplot as plt

# Path to the log file
log_path = 'training_log'

# Lists to store epoch, training loss, and validation loss
epochs = []
train_losses = []
val_losses = []

# Regular expressions to match the relevant lines
epoch_re = re.compile(r"Epoch (\d+)/\d+ Summary:")
train_loss_re = re.compile(r"Avg Training Loss: ([0-9.]+)")
val_loss_re = re.compile(r"Avg Validation Loss: ([0-9.]+)")

with open(log_path, 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        epoch_match = epoch_re.search(lines[i])
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            # Look ahead for loss lines
            train_loss = None
            val_loss = None
            for j in range(1, 5):
                if i + j < len(lines):
                    t_match = train_loss_re.search(lines[i + j])
                    v_match = val_loss_re.search(lines[i + j])
                    if t_match:
                        train_loss = float(t_match.group(1))
                    if v_match:
                        val_loss = float(v_match.group(1))
            if train_loss is not None and val_loss is not None:
                epochs.append(epoch_num)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
        i += 1

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.eps', format='eps')
plt.savefig('loss_curve.png', format='png')
print('Saved loss curve as loss_curve.eps')
