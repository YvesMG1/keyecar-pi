# Results

The repo includes four models. All see the last three camera frames and are trained on the same
data; [approach.md](approach.md#training) lists how they differ. The numbers come from
`python -m keyecar.training.eval` on 4 validation sessions of my own driving (8,453 frames) and on
one correction session that is left out of training.

Unless stated otherwise, each model gets its own last output as the previous command, as on the car.
Steering ranges from -1 to 1, so an error of 0.1 is 5% of the full range.

The models have been evaluated offline only. They have not driven on the car yet.

## Steering and throttle

| | Copy | Standard | No previous command | Backbone fully trained | Steering as bins |
|---|---:|---:|---:|---:|---:|
| Steering error (MAE) | 0.026 | 0.112 | 0.117 | 0.115 | **0.100** |
| Steering error, strong steering | 0.023 | 0.089 | 0.087 | 0.089 | **0.071** |
| Steering error, changed frames | 0.223 | 0.200 | 0.200 | **0.193** | 0.208 |
| Throttle error (MAE) | | **0.040** | 0.044 | 0.042 | 0.044 |
| Steering change per frame | | **0.024** | 0.047 | 0.045 | 0.049 |

*Copy* repeats my previous command without looking at the image. It needs my command from one frame
earlier, which no model has on the car, so it is a reference and not a competitor. *Changed frames*
are the 303 frames (3.6%) where my command moves more than 0.15 away from the previous one.

- Steering as bins has the lowest steering error. The other three are within noise of each other.
- On changed frames, where the image has to supply the answer, all four are only a little better
  than copying.
- My own commands change by 0.026 per frame. The standard model is as smooth as that, and the models
  without the previous command are about twice as jittery.

## Following my steering

*Steering steps* are the 106 moments where I go from a steady command to one at least 0.4 away.
*Whole turns* are the 79 turns that lie completely inside an unbroken stretch of frames. A turn
counts as followed if the model's summed steering is between 80% and 125% of mine, whenever exactly
it steers.

| | Standard | No previous command | Backbone fully trained | Steering as bins |
|---|---:|---:|---:|---:|
| Steering steps, median delay | 66 ms | 0 ms | 0 ms | 64 ms |
| 90% of steering steps followed within | 264 ms | 186 ms | 200 ms | **161 ms** |
| Steering steps not followed within 1 s | 1.9% | 1.9% | 0.9% | 0.9% |
| Share of the steering step reached | 90% | 100% | 97% | 98% |
| Whole turns followed | 77% | 84% | 81% | **89%** |
| Whole turns understeered | 18% | 16% | 15% | **9%** |
| Summed steering in a turn, median share of mine | 96% | 96% | 95% | 99% |

One frame is 64 ms, so a median delay of 0 or 64 ms means the models turn in with me or one frame
later. Over a whole turn the models typically apply 95% to 99% of my summed steering, and where a
turn is not followed, the model has mostly understeered. Steering as bins follows the most turns.

## A held-out correction session

The validation sessions are my own driving, where the car is never off the track. The held-out
correction session has 133 takeovers the models have not seen. The measure is taken on the first
frame of each takeover: how much of the gap between the steering of the model that was driving and
my correction does the evaluated model close? 0 is no better, 1 is my correction.

| First frame of each takeover | Standard | No previous command | Backbone fully trained | Steering as bins |
|---|---:|---:|---:|---:|
| Gap closed (mean) | 0.44 | **0.47** | 0.42 | 0.46 |
| Moved toward my correction | **75%** | 73% | 74% | 71% |
| Closed at least half of the gap | 47% | **53%** | 47% | **53%** |
| Steering error (0.42 for the model that was driving) | 0.26 | **0.24** | 0.25 | 0.27 |

The corrections carry over about halfway to takeovers the models have not seen. The four models do
not differ here.

## The previous command as input

The standard model, by what it gets as the previous command:

| Previous command | Steering error (MAE) | Changed frames | Whole turns followed |
|---|---:|---:|---:|
| My recorded command | 0.051 | 0.174 | 99% |
| Its own last output | 0.112 | 0.200 | 77% |
| Zero | 0.113 | 0.191 | 80% |

My recorded command is not available on the car, and repeating it alone already scores 0.026. With
its own output the standard model scores the same as with the input set to zero, and the same as the
models trained without it. About 0.11 is what the models get from the image.

## Speed on the Pi 5

Measured during autonomous runs with the standard architecture, using FP32 ONNX Runtime on 4 CPU
threads: about 5 ms for the network and about 9 ms for the whole frame (median). The camera runs at
20 fps, so the car reacts to every frame. The other three models differ only in the small head.

## Limitations

- These are offline numbers on images from my own drives, so an error cannot grow over time as it
  does on the car. An on-track comparison, counting takeovers per minute, is still to do.
- The validation set is small (4 sessions, 106 steering steps, 79 turns, one held-out correction
  session), and each model was trained once. Differences of 0.01 in steering error, 0.05 in gap
  closed, or 5 points in turns followed are within noise.
- The models overfit. Without the previous command, the validation steering error stops improving
  after epoch 10 and stays between 0.10 and 0.15, while the training error keeps falling to between
  0.04 and 0.06. More varied data is the likely next step, more than a different model or loss.
