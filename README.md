# PiTorch — a chessboard analyzer built on a from-scratch neural network

Written in Rust, **without any machine-learning library**: the neural network (forward pass, backpropagation, training, serialization) is implemented from scratch on top of `ndarray` for matrix arithmetic.

Given a chessboard in [Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation),
the analyzer predicts the state of the game:

| Output | Meaning |
|---|---|
| `Nothing` | nothing in particular |
| `Check <colour>` | `<colour>` has the other player's king in check |
| `Checkmate <colour>` | `<colour>` has won |

The project has two parts:

- a **neural network module** (library): create, save/load and train networks, configurable per layer;
- a **chessboard analyzer** (`PiTorch_analyzer`) that trains a network or uses it to predict.

A third executable, `PiTorch_generator`, creates blank random networks from a configuration file.

---

## Build

```bash
make        # builds PiTorch_analyzer and PiTorch_generator at the repository root
make clean  # removes the cargo build directory
make fclean # clean + removes the two binaries
make re     # fclean + all
```

Requirements: a recent stable Rust toolchain (`cargo`). Python 3 with `matplotlib` is only needed for the plotting
script in `scripts/`.

## Usage

### 1. Generate a blank network

```
./PiTorch_generator config_file_1 nb_1 [config_file_2 nb_2 ...]
```

```bash
$ ./PiTorch_generator basic_network.conf 3 && ls
basic_network_1.nn  basic_network_2.nn  basic_network_3.nn  basic_network.conf  PiTorch_generator
```

### 2. Train

```
./PiTorch_analyzer --train [--save SAVEFILE] LOADFILE CHESSFILE
```

Every line of `CHESSFILE` is a FEN followed by the expected output (`Nothing`, `Check White`, `Checkmate Black`, ...).
The trained network is written to `SAVEFILE`, or back into `LOADFILE` if `--save` is not given. Per-epoch metrics
are written to `training_log.csv` (epoch, train loss, validation loss, validation accuracy).

```bash
cat dataset/*/*.txt > all.txt
./PiTorch_analyzer --train --save PiTorch_network_v1.nn basic_network_1.nn all.txt
```

### 3. Predict

```
./PiTorch_analyzer --predict LOADFILE CHESSFILE
```

One answer per line of `CHESSFILE`, in the same order. An expected output may follow the FEN and is ignored.

```bash
$ ./PiTorch_analyzer --predict PiTorch_network_v1.nn chessboards.txt | head -n 5
Check White
Nothing
Nothing
Check White
Nothing
```

Errors are written on the error output and the program exits with code **84** (0 when everything went well).

### Configuration file (TOML)

```toml
n_inputs = 769

[[layer]]
units = 256
activation = "leaky_relu"
alpha = 0.01          # optional, only for leaky_relu and elu

[[layer]]
units = 64
activation = "relu"
init = "He"           # optional: xavier | he | lecun (default depends on the activation)

[[layer]]
units = 3
activation = "softmax"
```

Available activations: `linear`, `relu`, `leaky_relu`, `elu`, `selu`, `sigmoid`, `tanh`, `softplus`, `softmax`.
An invalid `alpha` (negative or not finite) is an error; an omitted `alpha` uses the documented default
(0.01 for Leaky ReLU, 1.0 for ELU). The analyzer expects `n_inputs = 769` and a last layer of 3 units.

---

## Project layout

```
.
├── Makefile
├── Cargo.toml
├── src/
│   ├── bin/
│   │   ├── PiTorch_analyzer.rs     # train / predict executable
│   │   └── PiTorch_generator.rs    # blank network generator
│   └── lib/                         # the neural network module
│       ├── activation.rs            # activation functions and their derivatives
│       ├── layer.rs                 # a layer = weights + biases + activation, weight initialization
│       ├── loss.rs                  # cross-entropy, MSE, MAE, Huber
│       ├── network.rs               # forward pass, backpropagation, gradient step
│       ├── io.rs                    # network file format (save / load)
│       ├── config.rs                # TOML configuration -> network
│       ├── trainer.rs               # mini-batch training loop, validation, early stopping
│       ├── metrics.rs               # accuracy, confusion matrix, precision / recall
│       └── chess/                   # FEN parsing, labelled-file parsing, mini-batch construction
├── scripts/                         # plotting and evaluation helpers
├── docs/                            # benchmark plots
└── PiTorch_network_*.nn            # pre-trained network(s)
```

---

## Design choices

### Input encoding (769 inputs)

A board is encoded as 12 piece types × 64 squares (one-hot) plus one input for the side to move.
Castling rights, en passant and the move counters are deliberately left out: they do not change whether a king
is in check or mated, and fewer useless inputs means fewer weights to overfit.
Each board has at most 33 active inputs, so samples are stored as lists of active indices and the dense matrix is
only built one mini-batch at a time (the full dataset as dense floats would need about 2.2 GB).

### Output and colour

The network has **3 outputs** (`Nothing`, `Check`, `Checkmate`) with a softmax. The colour in the dataset labels is
always the side that is *not* to move (the side to move is the one in check), in 100% of the 423,019 coloured
lines. It is therefore computed from the FEN when printing the answer instead of being learned, which removes
two output classes and a source of errors.

### Neural network module

- **Per-layer configuration**: every layer has its own size, activation and weight initialization.
- **Forward pass**: `a = f(x·W + b)`, with mini-batches stored as matrix rows. Activations are applied in place
  and each layer output is stored once during training (no duplicated caches).
- **Backpropagation**: weight gradient `dW = inputᵀ·δ / batch`, bias gradient `db = Σδ / batch`.
  For softmax + cross-entropy the output error is the fused, numerically stable `out − y`. Any other
  loss/activation combination, including softmax in a hidden layer, goes through the general path (loss gradient,
  then the activation derivative; the softmax Jacobian is applied as `s · (g − g·s)` per row).
  The implementation was validated against numerical (finite-difference) gradients.
- **Activations**: ReLU, Leaky ReLU, ELU, SELU, Softplus, sigmoid, tanh, linear, softmax.
- **Weight initialization**: uniform on `(−a, a)` with `a = √(3·Var)`, where the target variance is
  `1/n_in` (LeCun), `2/n_in` (He) or `2/(n_in+n_out)` (Xavier). The default follows the activation
  (He for the ReLU family and Softplus, LeCun for SELU, Xavier otherwise).
- **Losses**: cross-entropy, MSE, MAE and Huber (selectable; cross-entropy is used for classification).
- **Optimization**: mini-batch stochastic gradient descent.

### Avoiding overfitting

- Positions are deduplicated before training, so the same board cannot be in both the training and validation sets.
- A random validation split (10%) is held out; the validation loss is evaluated after every epoch.
- **Early stopping** with a patience of 3 epochs; the weights of the best epoch are restored at the end.

Training defaults (`TrainConfig::default()`): learning rate 0.05, batch size 256, up to 30 epochs,
cross-entropy loss. These are starting points, tuned with the benchmarks below.

### Network file format

A single binary file, little-endian:

| Bytes | Content |
|---|---|
| 8 | magic `MYTORCH1` (format signature and version) |
| 4 | length `N` of the header (u32) |
| `N` | JSON header: number of inputs, and per layer its number of units and activation |
| rest | for each layer: weights (f32, row-major), then biases (f32) |

`load` checks the signature, the header, the exact size of the file and the shape of every layer, and reports an
error (exit code 84) for a corrupt or truncated network.

### Dataset

The labelled dataset is provided with the subject and is **not** committed (too large). It contains 723,019
boards in 9 files (`nothing/`, `check/`, `checkmate/`, each with `10_pieces`, `20_pieces` and `many_pieces`).

| Class | Boards | Share |
|---|---|---|
| Nothing | 300,000 | 41.5% |
| Check | 249,300 | 34.5% |
| Checkmate | 173,719 | 24.0% |

7,464 positions appear more than once (no conflicting labels); they are removed for training. To rebuild the
training file: `cat dataset/*/*.txt > all.txt`.

---

## Benchmarks

> **TODO — fill in with your own runs.** A benchmark without a plot is not worth anything, so every row below
> should have a curve in `docs/`.

### Reproducing the plots

`training_log.csv` is overwritten by every training run, so copy it after each experiment
(for example `cp training_log.csv logs/arch_a.csv`). Then save the following as `scripts/plot_curves.py`:

```python
import csv
import sys

import matplotlib.pyplot as plt


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows],
            [float(r["val_loss"]) for r in rows], [float(r["val_acc"]) for r in rows])


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
for path in sys.argv[1:]:
    epochs, train_loss, val_loss, val_acc = read(path)
    name = path.rsplit("/", 1)[-1].removesuffix(".csv")
    ax_loss.plot(epochs, train_loss, "--", label=f"{name} (train)")
    ax_loss.plot(epochs, val_loss, label=f"{name} (validation)")
    ax_acc.plot(epochs, val_acc, label=name)

ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy")
ax_loss.legend()
ax_acc.legend()
fig.tight_layout()
fig.savefig("docs/learning_curves.png", dpi=150)
```

```bash
python3 scripts/plot_curves.py logs/arch_a.csv logs/arch_b.csv
```

### Measuring accuracy on a held-out test file

```bash
cut -d' ' -f7- test.txt > expected.txt
./PiTorch_analyzer --predict PiTorch_network_v1.nn test.txt > predicted.txt
paste -d'|' expected.txt predicted.txt | awk -F'|' '$1==$2 {ok++} END {print ok/NR}'
```

### Results

| Architecture | Activations | Loss | Epochs run | Val. accuracy | Recall Nothing / Check / Checkmate |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

*Learning curves:* `docs/learning_curves.png` (TODO)

*Confusion matrix of the pre-trained network:* TODO

*Accuracy per piece count (`10_pieces` / `20_pieces` / `many_pieces`):* TODO

*Comparison of hyperparameters (learning rate, batch size, depth/width, activation, loss):* TODO

---

## Known limitations and future work

- Some `Check` / `Checkmate` boards are still misclassified; more training and larger networks are being explored.
- Stalemates are not handled (bonus).
- Only plain mini-batch SGD is implemented; momentum / Adam, dropout and L2 regularization are not yet available.
- Training settings (learning rate, batch size, epochs, loss) are defaults in code and are not yet stored in the
  network file or the configuration file.