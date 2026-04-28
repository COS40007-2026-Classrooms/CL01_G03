"""
Simple Linear Regression Model with TensorFlow
A minimal example demonstrating regression with TensorFlow/Keras
"""

# Suppress TensorFlow warnings and info messages
import os
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Show only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD environments
import matplotlib.pyplot as plt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plot training data, test data, and model predictions.
    """
    plt.figure(figsize=(8, 6))

    plt.scatter(train_data, train_labels, c='b', label='Training data', alpha=0.7)
    plt.scatter(test_data, test_labels, c='g', label='Testing data', alpha=0.7)
    plt.scatter(test_data, predictions, c='r', label='Predictions', alpha=0.7)

    plt.title('Model Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.xlabel('X values', fontsize=12)
    plt.ylabel('Y values', fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_results.png', dpi=120, bbox_inches='tight')
    plt.close()


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MAE and MSE using NumPy.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mae), float(mse)


def save_metrics_to_file(mae, mse, filename='metrics.txt'):
    """
    Save metrics to a text file.
    """
    with open(filename, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write("=" * 50 + "\n")
    print(f"Metrics saved to {filename}")


# ============================================================================
# DATA PREPARATION
# ============================================================================

print(f"TensorFlow version: {tf.__version__}")

# Create synthetic data (linear relationship: y = x + 10)
X = np.arange(-100, 100, 4)   # 50 samples
y = np.arange(-90, 110, 4)    # 50 samples

print(f"Dataset size: {len(X)} samples")

# Reshape to 2D (samples, features)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split data: 80% train, 20% test
split_idx = 40
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test  = X[split_idx:]
y_test  = y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# ============================================================================
# MODEL BUILDING
# ============================================================================

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), name='linear_layer')
])

model.compile(
    loss='mae',
    optimizer='sgd',
    metrics=['mae']
)

print("\nModel Architecture:")
model.summary()

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "=" * 50)
print("TRAINING MODEL")
print("=" * 50)

history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,
    validation_split=0.2
)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

y_preds     = model.predict(X_test, verbose=0).flatten()
y_test_flat = y_test.flatten()

plot_predictions(
    X_train.flatten(), y_train.flatten(),
    X_test.flatten(),  y_test_flat,
    y_preds
)

mae_value, mse_value = calculate_metrics(y_test_flat, y_preds)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")
print(f"Mean Squared Error (MSE):  {mse_value:.2f}")
print("=" * 50)

save_metrics_to_file(mae_value, mse_value)

# ============================================================================
# TRAINING HISTORY PLOT
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'],     label='Training Loss',   linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss (MAE)', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'][10:],     label='Training Loss',   linewidth=2)
axes[1].plot(history.history['val_loss'][10:], label='Validation Loss', linewidth=2)
axes[1].set_title('Loss (Epochs 10+)', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE MODEL
# ============================================================================

model.save('linear_regression_model.h5')
print("\nModel saved as 'linear_regression_model.h5'")

sample_input      = np.array([[50.0]])
sample_prediction = model.predict(sample_input, verbose=0)
print(f"\nExample prediction: X={sample_input[0][0]:.0f} -> y={sample_prediction[0][0]:.2f}")

weights = model.get_weights()
if len(weights) >= 2:
    print(f"\nWeight (slope):    {weights[0][0][0]:.4f}")
    print(f"Bias (intercept):  {weights[1][0]:.4f}")
    print(f"Expected:          y = x + 10")
    print(f"Learned:           y = {weights[0][0][0]:.4f}*x + {weights[1][0]:.4f}")

print("\n" + "=" * 50)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("=" * 50)
