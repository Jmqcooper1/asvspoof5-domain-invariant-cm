# OOD Gap Analysis: Dev vs Eval Generalization

| Model | Backbone | Dev EER (%) | Eval EER (%) | Gap | Gap Reduction |
| --- | --- | --- | --- | --- | --- |
| ERM | WavLM | 3.26 | 8.48 | 5.22 | (baseline) |
| DANN | WavLM | 4.76 | 7.36 | 2.61 | 50.1\% |
| ERM | W2V2 | 4.24 | 15.15 | 10.91 | (baseline) |
| DANN | W2V2 | 4.45 | 14.37 | 9.92 | 9.2\% |
| DANN v2 | W2V2 | 7.81 | 18.54 | 10.73 | 1.7\% |