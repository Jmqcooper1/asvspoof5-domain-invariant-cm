# OOD Gap Analysis: Dev vs Eval Generalization

| Model | Backbone | Dev EER (%) | Eval EER (%) | Gap | Gap Reduction |
| --- | --- | --- | --- | --- | --- |
| ERM | WavLM | 3.81 | 8.48 | 4.68 | (baseline) |
| DANN | WavLM | - | 7.36 | - | - |
| ERM | Wav2Vec2 | - | 15.15 | - | (baseline) |
| DANN v1 | Wav2Vec2 | - | 14.37 | - | - |
| DANN v2 | Wav2Vec2 | - | 18.54 | - | - |
| GMM | LFCC | 17.59 | 43.33 | 25.74 | - |
| Logistic | TRILLsson | 19.35 | 23.75 | 4.40 | - |
| MLP | TRILLsson | 20.32 | 25.65 | 5.33 | - |