# Datasets for the QC-CNN-Parallel Experiment

> **Cross-verification note:** This file has been updated against the paper PDF
> *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized
> Quantum Circuits for Image Classification*, Quantum Engineering (2026),
> article 6643049. All paper-reported splits and dataset properties now match
> Table 1 and Section 4.2.1 of the paper.

## 0. Datasets actually used in the paper (verified from PDF)

The paper uses exactly **three grayscale 28 × 28 datasets** (Table 1, page 7):

| Dataset | All classes | Image size | Training samples | Test samples | Source |
|---|---|---|---:|---:|---|
| MNIST | All 10 classes | 28 × 28 × 1 | 10,000 (1,000/class) | 2,000 (200/class) | Class-balanced subsampling |
| Fashion-MNIST | All 10 classes | 28 × 28 × 1 | 10,000 (1,000/class) | 2,000 (200/class) | Class-balanced subsampling |
| Overhead-MNIST | All classes | 28 × 28 × 1 | 8,519 | 1,065 | Full dataset used |

**Important clarification from Section 4.2.1 of the paper:**
- For MNIST and Fashion-MNIST, all ten categories were retained. **1,000 samples per
  category** were randomly selected for training and **200 samples per category** for
  testing. This is class-balanced subsampling—not the full 60,000/10,000 split.
- For Overhead-MNIST, the full dataset was used as-is because its original scale
  is already suitable for experimentation.
- The paper explicitly states this subsampling was done because of "computational
  limitations of current quantum simulators."

## 1. Dataset currently represented by the code

The file `qc-cnn-parallel.py` does not currently load a real dataset. It creates
random dummy images:

```python
dummy_images = torch.rand((batch_size, 1, 28, 28))
dummy_labels = torch.randint(0, 10, (batch_size,))
```

This shows that the intended experiment is designed around an MNIST-like
dataset:

| Property | Current model assumption |
|---|---:|
| Image type | Grayscale |
| Input channels | 1 |
| Image size | 28 x 28 pixels |
| Number of classes | 10 by default |
| Input tensor | `[batch, 1, 28, 28]` |
| Pixel range | Preferably `[0, 1]` |

The original MNIST dataset contains 60,000 training images and 10,000 test
images. Each image is a centered, size-normalized 28 x 28 grayscale image of a
handwritten digit. However, **the paper only used 10,000 training and 2,000 test
images** (1,000 and 200 per class respectively) due to quantum simulator
computational constraints. See the [official MNIST database page](https://yann.lecun.com/exdb/mnist/).

### Recommended baseline: MNIST (paper-matched configuration)

MNIST should be the baseline dataset because it matches the current model with
no architectural changes:

```text
MNIST image [1, 28, 28]
        |
2 x 2 quantum patches with stride 2
        |
14 x 14 patch grid
```

The ten labels are digits `0` through `9`, so the model can remain:

```python
model = QCCNNParallel(num_classes=10)
```

**Paper-matched training split:** To faithfully reproduce the paper, use
1,000 samples per class (10,000 total) for training and 200 samples per class
(2,000 total) for testing—not the full MNIST dataset.

## 2. Dataset loading specification

For a real experiment, replace the dummy data with a reproducible train/valid/
test pipeline. A PyTorch implementation can use:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),                 # [0, 255] -> [0, 1]
])

train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

The model should receive images with shape `[B, 1, 28, 28]`. Labels should be
integer class IDs with shape `[B]`, which is the expected format for
`torch.nn.CrossEntropyLoss`.

## 3. Stronger drop-in alternative: Fashion-MNIST

**Fashion-MNIST is the best first advanced dataset for this model.** It is a
direct replacement for MNIST: it has 70,000 28 x 28 grayscale images in 10
classes, with 60,000 training images and 10,000 test images. The classes are
fashion products such as shirts, trousers, coats, shoes, and bags. The dataset
has the same image size and split structure as MNIST, but is substantially more
challenging than handwritten digits.

Sources:

- [Fashion-MNIST paper](https://arxiv.org/abs/1708.07747)
- [Official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist)

Only the dataset class needs to change:

```python
train_dataset = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)
```

The model remains unchanged:

```python
model = QCCNNParallel(num_classes=10)
```

### Why Fashion-MNIST is suitable

- Same `[1, 28, 28]` input shape.
- Same ten-class output structure.
- Same quantum patch extraction and dense-layer dimensions.
- More meaningful test of feature extraction than MNIST.
- Easy comparison with classical CNN and quantum-classical baselines.

## 4. More advanced character dataset: EMNIST Balanced

EMNIST extends handwritten digit recognition to handwritten characters. The
EMNIST Balanced split contains 131,600 examples across 47 balanced classes and
uses the same 28 x 28 image format and MNIST-like structure. The official
dataset information is available from [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).

This is a stronger handwriting experiment than MNIST, but the classifier must
be changed:

```python
model = QCCNNParallel(num_classes=47)
```

The convolutional and quantum branches do not need to change because the input
remains grayscale 28 x 28. Only the final output layer and label handling need
to reflect the selected EMNIST split.

## 5. Third paper dataset: Overhead-MNIST

**Overhead-MNIST is the third dataset actually used in the paper** (Table 1,
page 7). It is a remote-sensing image classification dataset that has been
resized to 28 × 28 grayscale, making it directly compatible with the current
model architecture. The classes include aerial/overhead objects such as:
Car, Harbor, Helicopter, Oil_gas_field, Parking_lot, Plane, Runway_mark,
Ship, Stadium, and Storage_tank (visible in Figure 1 of the paper).

Sources:

- [Official Overhead-MNIST repository](https://github.com/reveondivad/ov-mnist)

The paper used the **full Overhead-MNIST dataset**: 8,519 training images and
1,065 test images (no subsampling applied). Because the class count may differ
from 10, set `num_classes` from the dataset metadata:

```python
model = QCCNNParallel(num_classes=num_overhead_classes)
```

The convolutional and quantum branches do not need to change because the input
remains grayscale 28 × 28. Only the final output layer needs to match the
Overhead-MNIST class count.

## 5b. Recommended research-oriented dataset: MedMNIST

MedMNIST is a suitable advanced option if the M.Tech research is meant
to demonstrate a biomedical application. It provides standardized biomedical
image classification datasets, including grayscale and RGB 2D subsets. The
standard 2D version is resized to 28 x 28, which makes it compatible with the
current quantum patch design. The collection includes different task types,
including binary, multi-class, ordinal, and multi-label classification.

Sources:

- [MedMNIST official website](https://medmnist.com/)
- [MedMNIST official repository](https://github.com/MedMNIST/MedMNIST)
- [MedMNIST v2 research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9852451/)

Examples of suitable 2D subsets include `pathmnist`, `bloodmnist`, and
`dermamnist`. The exact number of classes depends on the chosen subset, so set
`num_classes` from the dataset metadata rather than hard-coding 10.

Important research caution: MedMNIST images are standardized and resized for
benchmarking. They are appropriate for an ML experiment, but the model's
results should not be interpreted as a clinical diagnostic system. Follow the
dataset license, source-dataset terms, and any required ethics or data-use
requirements.

## 6. More difficult natural-image option: CIFAR-10

CIFAR-10 contains 60,000 32 x 32 colour images in 10 classes, split into 50,000
training images and 10,000 test images. The official dataset page is hosted by
the [University of Toronto](https://cave.cs.toronto.edu/kriz/cifar.html).

CIFAR-10 is more advanced than MNIST or Fashion-MNIST, but it is **not a direct
drop-in replacement** for the current implementation because:

- The images have 3 colour channels instead of 1.
- The images are 32 x 32 instead of 28 x 28.
- The current quantum circuit consumes only four scalar values from a 2 x 2
  grayscale patch.
- The current dense layer is fixed to `12 * 14 * 14 = 2352` inputs.

To use CIFAR-10, choose one of these experimental designs:

1. Convert RGB images to grayscale and resize them to 28 x 28. This preserves
   the current architecture but discards colour information.
2. Change the classical convolution to `in_channels=3`, redesign the quantum
   encoding to handle colour, and recalculate all output dimensions.
3. Use separate quantum encodings for the RGB channels. This is a new model
   variant and should be reported as an architectural modification.

For a first thesis experiment, Fashion-MNIST or Overhead-MNIST is
preferable because they are the datasets the paper itself uses.

## 7. Dataset recommendation for the thesis experiments

**To faithfully reproduce the paper**, use exactly the three paper datasets:

| Stage | Dataset | Paper split | Purpose | Model changes |
|---|---|---|---|---|
| Baseline | MNIST | 10,000 train / 2,000 test (1,000 & 200 per class) | Validate implementation | None |
| Main benchmark | Fashion-MNIST | 10,000 train / 2,000 test (1,000 & 200 per class) | Test harder dataset | None |
| Application benchmark | Overhead-MNIST | 8,519 train / 1,065 test (full dataset) | Remote sensing application | Set `num_classes` |

**For extended experiments beyond the paper:**

| Stage | Dataset | Purpose | Model changes |
|---|---|---|---|
| Optional stress test | EMNIST Balanced | Test many-class handwritten recognition | Set `num_classes=47` |
| Biomedical application | One grayscale MedMNIST subset | Demonstrate biomedical relevance | Set `num_classes`; inspect labels |

The strongest practical choice for paper reproduction is to follow the three-
dataset structure from the paper: MNIST (digit baseline), Fashion-MNIST
(harder texture-based), and Overhead-MNIST (real-world aerial imagery).

## 8. Required experiment reporting

For every dataset, record:

- Dataset name, version, and official source.
- Number of training, validation, and test samples.
- Image dimensions and number of channels.
- Number of classes and class names.
- Normalization and resizing operations.
- Random seed and data split method.
- Batch size, optimizer, learning rate, and number of epochs.
- Test accuracy, macro-F1, confusion matrix, and test loss.
- Classical-CNN baseline with comparable parameter budget.
- QC-CNN-Parallel runtime and number of trainable quantum parameters.

Do not compare accuracy values from different datasets as if they measured the
same task. Compare the quantum model against an equivalent classical baseline
on the same fixed split.

## 9. Dataset-specific compatibility summary

| Dataset | Input | Classes | Drop-in compatible? | Paper dataset? | Recommendation |
|---|---|---:|---|---|---|
| MNIST | 1 x 28 x 28 grayscale | 10 | Yes | **Yes** (Table 1) | Baseline (use 10,000 train / 2,000 test as in paper) |
| Fashion-MNIST | 1 x 28 x 28 grayscale | 10 | Yes | **Yes** (Table 1) | Main benchmark (same split as MNIST) |
| Overhead-MNIST | 1 x 28 x 28 grayscale | ~11 aerial classes | Yes, set `num_classes` | **Yes** (Table 1) | Remote-sensing benchmark (full dataset: 8,519 / 1,065) |
| EMNIST Balanced | 1 x 28 x 28 grayscale | 47 | Yes, change output classes | No | Optional stress test |
| MedMNIST 2D grayscale subset | Usually 1 x 28 x 28 | Dataset-dependent | Usually, inspect subset | No | Biomedical extension |
| CIFAR-10 | 3 x 32 x 32 RGB | 10 | No | No | Later architecture extension |
