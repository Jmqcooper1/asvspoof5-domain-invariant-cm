# Main Results: EER and minDCF for ERM vs DANN

| Model | Backbone | Dev EER (%) | Eval EER (%) | Eval minDCF |
| --- | --- | --- | --- | --- |
| ERM | WavLM | 3.81 | 8.48 | 0.6360 |
| DANN | WavLM | - | 7.36 | 0.5769 |
| ERM | Wav2Vec2 | - | 15.15 | 0.9997 |
| DANN v1 | Wav2Vec2 | - | 14.37 | 0.9947 |
| DANN v2 | Wav2Vec2 | - | 18.54 | 1.0000 |
| GMM | LFCC | 17.59 | 43.33 | 0.9995 |
| Logistic | TRILLsson | 19.35 | 23.75 | 1.0000 |
| MLP | TRILLsson | 20.32 | 25.65 | 1.0000 |