# Self-Pruning Neural Network Report

## Explanation

The model uses sigmoid gates on each weight. Sigmoid outputs values between 0 and 1.

The sparsity loss is the mean of all gate values. Minimizing it pushes gates toward 0.

When a gate is near 0, it multiplies the weight by ~0, so the weight has no effect. This removes it from the network.

The network learns to keep gates open only for useful weights. Useless weights get their gates closed. This is how it prunes itself during training.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.0    | 51.91       | 0.00         |
| 1.0    | 52.27       | 1.92         |
| 5.0    | 53.12       | 28.48        |

## Analysis

With lambda = 0, there is no sparsity penalty. All gates stay open. No pruning happens.

As lambda increases, the penalty pushes more gates toward 0. Sparsity goes up.

Accuracy stays roughly the same across all three runs. This means the pruned weights were not useful. Removing them did not hurt the model.

This shows the trade-off: higher lambda gives more pruning with little accuracy cost.

## Gate Distribution Plot

The plot shows all gate values from the best model (lambda = 5.0).

There is a spike near 0. These are the pruned weights — their gates were pushed closed by the sparsity loss.

Some gates have high values, close to 1. These are the weights the network decided to keep.

The two groups show the model learned to separate useful weights from useless ones.

## Conclusion

The model successfully prunes itself during training. No manual pruning was needed.

Higher lambda means more pruning. Lower lambda means the network stays dense.

Accuracy does not drop much even with 28% sparsity. The method works.