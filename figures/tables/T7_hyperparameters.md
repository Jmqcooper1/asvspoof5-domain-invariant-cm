# Training Hyperparameters

| Category | Parameter | Value |
| --- | --- | --- |
| Architecture | Backbone | WavLM Base+ / W2V2 Base |
|  | Backbone layers | 12 (frozen) |
|  | Projection dim | 256 |
|  | Classifier hidden | 256 |
|  | Domain head hidden | 256 |
| Training | Batch size | 32 |
|  | Learning rate | 1e-4 |
|  | Optimizer | AdamW |
|  | Weight decay | 0.01 |
|  | Epochs | 10 |
|  | Early stopping | Patience 3 |
| DANN | $\\lambda$ schedule | Linear ramp 0→1 |
|  | GRL | Gradient reversal layer |
|  | Domain targets | CODEC + CODEC\_Q |
| Regularization | Dropout | 0.1 |
|  | Label smoothing | 0.0 |