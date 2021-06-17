# Person Re-identification
Final project of Computer Vision

## Train model
Download the Market-1501 dataset in [here](https://drive.google.com/file/d/1aCUQi3wCxaumbQCiI6xUOGsoY9rkENBa/view?usp=sharing)

Change `config.json` file if needed.

Run `main.py` to start training.

## Result

| Methods                                  | Train feature | Test feature | Rank-1 | Rank-5 |  mAP  |
|------------------------------------------|:-------------:|:------------:|:------:|:------:|:-----:|
| Baseline                                 |     Global    |    Global    |  83.28 |  94.18 | 65.96 |
| Baseline                                 |     Local     |     Local    |  80.20 |  92.76 | 59.66 |
| Baseline                                 | Global, Local |    Global    |  84.14 |  93.97 | 67.58 |
| Baseline                                 | Global, Local |     Local    |  86.37 |  95.10 | 68.68 |
| Baseline + Identity Loss                 | Global, Local |    Global    |  83.55 |  93.35 | 67.66 |
| Baseline + Identity Loss                 | Global, Local |     Local    |  84.23 |  93.62 | 63.99 |
| Baseline + Identity Loss + Random Flip   | Global, Local |    Global    |  85.42 |  93.97 | 69.34 |
| Baseline + Identity Loss + Random Flip   | Global, Local |     Local    |  86.82 |  94.89 | 67.59 |
| Re-ranking                               | Global, Local |    Global    |  87.47 |  92.96 | 82.90 |
| Re-ranking + Identity Loss               | Global, Local |    Global    |  86.19 |  92.76 | 81.27 |
| Re-ranking + Identity loss + Random Flip | Global, Local |    Global    |  87.50 |  92.93 | 82.68 |