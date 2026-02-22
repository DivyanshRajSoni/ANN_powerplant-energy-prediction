# Power Plant Energy Output Prediction using ANN

A PyTorch-based Artificial Neural Network (ANN) for predicting the electrical energy output of a Combined Cycle Power Plant using regression.

## Dataset

The dataset (`powerplant_data.csv`) contains **9568 observations** with the following features:

| Feature | Description |
|---------|-------------|
| **AT** | Ambient Temperature |
| **V** | Exhaust Vacuum |
| **AP** | Ambient Pressure |
| **RH** | Relative Humidity |
| **PE** | Produced Energy (Target) |

## Model Architecture

A feedforward ANN built with `nn.Sequential`:

```
Input (4 features)
  → Linear(4, 6) → ReLU
  → Linear(6, 6) → ReLU
  → Linear(6, 1)
Output (Energy Prediction)
```

## Workflow

1. **Data Loading** — Load CSV with Pandas
2. **Preprocessing** — Train/test split (80/20), StandardScaler normalization
3. **Tensor Conversion** — NumPy arrays → PyTorch tensors, wrapped in `DataLoader` (batch size 32)
4. **Training** — 100 epochs, Adam optimizer, MSE loss with best model checkpointing
5. **Evaluation** — MSE and R² score on test set
6. **Visualization** — Training vs. validation loss curves

## Requirements

- Python 3.x
- PyTorch
- Pandas
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Usage

Open and run `ANN_Regression.ipynb` in Jupyter Notebook or VS Code. The best model weights are saved to `best_model.pt` during training.

## Project Structure

```
├── ANN_Regression.ipynb   # Main notebook with full pipeline
├── powerplant_data.csv    # Dataset
├── best_model.pt          # Saved best model weights
└── README.md
```
