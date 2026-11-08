# Results

Both included models were trained with the same data and settings. The only difference is how many
camera frames they see. The numbers come from `python -m keyecar.training.eval` on 4 held-out
validation sessions (8,453 frames).

## Accuracy

Measured with the previous command taken from the recording, the only setting where the correct
answer is known. Steering ranges from -1 to 1, so an error of 0.05 is 2.5% of the full range.

| | Current frame | 3-frame history |
|---|---:|---:|
| Steering error (MAE) | **0.048** | 0.052 |
| Steering error, strong steering | **0.046** | 0.049 |
| Steering error, left turns | **0.053** | 0.059 |
| Steering error, right turns | **0.053** | 0.060 |
| Steering direction correct | 99.6% | **99.7%** |
| Throttle error (MAE) | 0.040 | **0.037** |

## Dependence on the previous command

The model's own last output replaces the recorded previous command, frame by frame, and the result is
compared with its normal prediction. This shows how much the model leans on its own output, which is
what goes wrong on the car (see [approach.md](approach.md#the-previous-command-problem)). Lower drift
and higher steering kept are better.

| | Current frame | 3-frame history |
|---|---:|---:|
| Average steering drift | 0.092 | **0.078** |
| Steering direction flips | 2.4% | **1.4%** |
| Steering kept (magnitude) | 94.9% | **99.4%** |

**Takeaway:** the single-frame model is slightly more accurate. The history model depends less on its
own previous output, which matters more when the car is driving itself, so `keyecar-drive` uses it
by default.

## Speed on the Pi 5

Measured during autonomous runs, using FP32 ONNX Runtime on 4 CPU threads:

| | Network only | Whole frame (median) |
|---|---:|---:|
| Current frame | ~4 ms | ~7.5 ms |
| 3-frame history | ~5 ms | ~9 ms |

The camera runs at 20 fps, so the car reacts to every frame.

## Limitations

- These are offline numbers. A proper on-track comparison of both models, counting takeovers per
  minute, is still to do. Offline accuracy did not predict on-track behaviour well during development.
- The validation set is small (4 sessions), and each model was trained only once.
