# Cross-Country Traffic Sign Classification

**Module M507 – Methods of Prediction · Final Assignment**

A convolutional neural network trained on **German** road signs (GTSRB) and evaluated on **unseen Belgian** road signs (BTSC). The goal is a classifier that generalizes to a different country, camera setup, lighting and background, rather than one that only scores well on its own training distribution.

| Result | Value |
|---|---|
| Belgian validation accuracy | 95.92% (loss 0.2195) |
| **Belgian test accuracy** | **97.78%** (loss 0.1123) |
| Balanced accuracy, all classes | 93.66% |
| Balanced accuracy, classes with ≥30 test samples | 97.91% |
| Macro / weighted F1 (test) | 0.94 / 0.98 |

## Business Context

Traffic sign recognition is a core component of Advanced Driver Assistance Systems (ADAS). A model that does well on its benchmark but fails on signs from a different setting is commercially useless and unsafe. This project therefore tests generalization directly, by training and evaluating on different countries' data.

## Task and Data

Multi-class image classification into **18 sign classes** shared by both datasets.

| Split | Source | Images |
|---|---|---|
| Train | [German GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) (80%) | 12,312 |
| German hold-out (sanity check only) | GTSRB (20%) | – |
| Validation | [Belgian BTSC](https://www.kaggle.com/datasets/abhi8923shriv/belgium-ts/) Training set | 1,667 |
| Test | Belgian BTSC Testing set | 854 |

Key data decisions:
- **Manual class mapping.** 19 candidate classes were shared between the datasets, and one was dropped because the Belgian test set had no images for it. The final 18 are intersection ahead, priority road, yield, stop, no vehicles, no vehicles (3.5t), no entry, general caution, curve left/right, double curve, bumpy road, slippery road, narrow road (right), road works, children, go straight and roundabout.
- **Format conversion.** Belgian `.ppm` files were converted to `.png` for the TensorFlow dataset pipeline.
- **Class imbalance.** Both datasets are imbalanced. In the Belgian test set about half the classes have fewer than 15 samples, which is why balanced accuracy is reported alongside accuracy.
- **Near-duplicate frames.** Images come from video keyframes, so a within-country split is optimistic. This was the reason for using a second country as the evaluation set.

## Model

A VGG-style CNN on 32×32 RGB input, inspired by Mishra & Goyal (2022). Convolutions are stacked in pairs before each pooling step. This gives more depth and a larger receptive field without shrinking the small feature map too early.

```
Rescaling(1/255)
3 × [ Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool ]   # 32 → 64 → 128 filters
Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(18, softmax)
```

- Adam (lr = 0.001), categorical cross-entropy, He-normal initialization
- Early stopping (patience 6, restore best weights) and ReduceLROnPlateau
- Fixed seed (36) for reproducibility
- Trained for 17 epochs in about 154 s

## Ablation Study

Ten single-change experiments were run against the final model. Every change lowered validation accuracy.

| # | Change | Val. Acc. | Val. Loss |
|:-:|---|---|---|
| – | **Final model** | **95.92%** | **0.2195** |
| 1 | No conv stacking (single conv per block) | 93.82% | 0.2754 |
| 2 | Different image size | 93.46% | 0.2795 |
| 3 | Augmentation (rotation + brightness) | 93.22% | 0.2778 |
| 4 | Remove dropout | 93.16% | 0.3026 |
| 5 | Global Average Pooling head | 92.50% | 0.2859 |
| 6 | Remove BatchNorm | 91.66% | 0.4787 |
| 7 | Grayscale input | 91.66% | 0.3096 |
| 8 | Reduce depth (3 → 2 blocks) | 90.52% | 0.3946 |
| 9 | Per-channel standardization instead of min-max scaling | 84.58% | 0.5971 |
| 10 | Learning rate 0.01 | 61.31% | 0.9834 |

Takeaways: the learning rate, input normalization and network depth mattered most, and color is informative for this task. Augmentation did not help in this setup.

## Evaluation

- **Cross-country generalization.** Test accuracy (97.78%) is at least as good as Belgian validation accuracy (95.92%).
- **Why two countries matter.** The same model scores **99.9%** on the held-out German split. That figure is inflated by near-duplicate frames and would have hidden the real generalization performance.
- **Per-class behavior.** Well-sampled classes are reliable. Rare classes swing widely, for example `curve_left` recall is 0.50 and `slippery_road` is 0.71, because a single error moves the score a lot when there are very few samples.
- **Balanced accuracy.** It is 93.66% over all classes and 97.91% when restricted to classes with ≥30 samples. The gap comes from tiny classes, which is a limitation of the test data and not necessarily of the model.

## Limitations and Recommendations

**Limitations**
- Small dataset with near-duplicate video frames.
- Very few samples in several classes, so per-class results are noisy.
- Only two countries were used.

**Recommendations**
- Collect more data for under-represented classes.
- Validate on more countries.
- **Not deployment-ready** because of the uneven performance on rare classes. It is a good base for fine-tuning on broader data.

## Tech Stack

Python · TensorFlow / Keras · OpenCV · scikit-learn · NumPy · Matplotlib · KaggleHub

## Running the Notebook

The notebook, `Methods_of_Prediction__Final_Assignment.ipynb`, was written for Google Colab. It downloads both datasets through `kagglehub`, so Kaggle access is needed. To run it locally:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy kagglehub
jupyter notebook Methods_of_Prediction__Final_Assignment.ipynb
```

## Reference

Mishra, J. and Goyal, S. (2022). *An effective automatic traffic sign classification and recognition deep convolutional networks.* Multimedia Tools and Applications, 81, 18915–18934. https://doi.org/10.1007/s11042-022-12531-w
