# Task 1: Classification Report — Pneumonia Detection with PneumoniaMNIST

## 1. Architecture Justification

### Modified UNet

The UNet architecture, originally designed for biomedical image segmentation, is adapted here for classification. Its encoder-decoder structure with skip connections preserves **spatial feature information** at multiple scales, which can be beneficial for detecting diffuse patterns in pneumonia (e.g., infiltrates, consolidation). The full UNet from scratch uses a symmetric encoder-decoder with skip connections followed by Global Average Pooling (GAP) and a fully connected layer. The pretrained variant uses a ResNet18 encoder backbone, leveraging ImageNet-learned features.

**Why it may help**: Pneumonia manifests as spatial patterns (opacities, ground-glass appearance) that benefit from multi-scale feature extraction. UNet's architecture preserves these spatial relationships better than purely classification-focused networks.

### ResNet18

ResNet18 uses residual connections that enable training of deeper networks without degradation. The skip connections allow gradients to flow more effectively, enabling the model to learn identity mappings when deeper features are not beneficial.

**Why it helps**: Residual connections solve the vanishing gradient problem, allowing the network to learn more complex hierarchical features. For medical images, this means the network can capture both low-level texture features and high-level semantic patterns associated with pneumonia.

### EfficientNet-B0

EfficientNet uses compound scaling to balance network width, depth, and resolution. EfficientNet-B0 achieves strong performance with significantly fewer parameters compared to other architectures.

**Why it helps**: Parameter efficiency is crucial in medical imaging where labeled data is limited. EfficientNet's mobile inverted bottleneck convolutions (MBConv) with squeeze-and-excitation blocks provide channel-wise attention, allowing the network to focus on the most informative features for pneumonia detection.

---

## 2. Training Methodology

### Hyperparameters

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Optimizer               | Adam             |
| Learning Rate           | 0.001            |
| Weight Decay            | 1e-4             |
| LR Scheduler            | ReduceLROnPlateau|
| Scheduler Patience      | 3 epochs         |
| Scheduler Factor        | 0.5              |
| Early Stopping Patience | 7 epochs         |
| Max Epochs              | 30               |
| Batch Size              | 64               |
| Random Seed             | 42               |

### Data Augmentation

- **Random Horizontal Flip** (p=0.5) — Chest X-rays can be mirrored without losing diagnostic meaning
- **Random Rotation** (±10°) — Accounts for minor positional variations
- **Random Affine** (translation=5%, scale=95-105%) — Simulates slight positioning changes
- **Color Jitter** (brightness/contrast=0.1) — Accounts for exposure variations

### Loss Function

**BCEWithLogitsLoss** — Binary cross-entropy with numerically stable logit input, appropriate for binary (Normal vs Pneumonia) classification.

### Normalization

Images normalized with mean=0.5, std=0.5 to center pixel values around zero. This is appropriate for medical imaging where consistent intensity scaling is important.

---

## 3. Experiment Results

> **Note**: The table below will be populated after running the experiment suite.
> Run `python -m task1_classification.experiment_runner` to generate results.

| Model        | Variant    | Parameters | Test Acc | Test Precision | Test Recall | Test F1 | Test AUC | Training Time |
| ------------ | ---------- | ---------- | -------- | -------------- | ----------- | ------- | -------- | ------------- |
| UNet         | Scratch    | —          | —        | —              | —           | —       | —        | —             |
| UNet         | Pretrained | —          | —        | —              | —           | —       | —        | —             |
| ResNet18     | Scratch    | —          | —        | —              | —           | —       | —        | —             |
| ResNet18     | Pretrained | —          | —        | —              | —           | —       | —        | —             |
| EfficientNet | Scratch    | —          | —        | —              | —           | —       | —        | —             |
| EfficientNet | Pretrained | —          | —        | —              | —           | —       | —        | —             |

Full results are exported to `reports/experiment_comparison.csv` after running experiments.

---

## 4. Analysis

### Pretrained vs. Scratch Performance

Pretrained models are expected to outperform scratch variants due to:
- **Transfer learning**: ImageNet features (edges, textures, shapes) transfer well to medical images
- **Faster convergence**: Pretrained weights provide a strong initialization
- **Better generalization**: Regularization effect of starting from well-learned feature representations

Even though ImageNet contains natural images, low-level features (Gabor-like filters, edge detectors) are universal and beneficial for medical image analysis.

### Expected Best Performing Model

ResNet18 (Pretrained) or EfficientNet-B0 (Pretrained) are expected to be top performers:
- ResNet's residual connections and proven track record on classification tasks
- EfficientNet's efficient architecture design balancing accuracy and parameters

### Overfitting Discussion

Key overfitting indicators to monitor:
- **Train-val gap**: Large difference suggests overfitting
- **Early stopping**: Monitors validation AUC to prevent overfitting
- **Data augmentation**: Regularization through augmentation helps reduce overfitting
- **Dropout**: Applied in classifier heads (0.3-0.5) for additional regularization

### Computational Cost Comparison

- **UNet Scratch**: Highest parameter count due to full encoder-decoder architecture
- **UNet Pretrained**: Leverages efficient ResNet18 backbone, fewer decoder parameters needed
- **ResNet18**: Moderate parameter count (~11M), efficient for its depth
- **EfficientNet-B0**: Most parameter-efficient (~5M), best accuracy/parameter ratio

---

## 5. Failure Case Analysis

> **Note**: Failure case visualizations are generated in `reports/` after running experiments.

Common patterns in misclassified pneumonia images:

- **Low contrast images**: Poor exposure makes opacities difficult to detect
- **Subtle/early pneumonia**: Small or diffuse infiltrates that even radiologists may disagree on
- **Normal variants**: Anatomical variations (e.g., prominent pulmonary vasculature) that mimic pneumonia
- **Image quality**: Rotation, cropping artifacts from the 28×28 downsampling
- **Borderline cases**: Images near the decision boundary where features are ambiguous

---

## 6. Conclusion

### Deployment Recommendation

For a clinical deployment scenario, the recommended model would be one that maximizes **recall** (sensitivity) — detecting all pneumonia cases is more important than minimizing false positives, as missed diagnoses carry higher risk than false alarms.

Key deployment considerations:
1. **High sensitivity priority**: In a screening context, false negatives are more costly
2. **Model size**: Smaller models enable edge deployment (mobile, embedded devices)
3. **Inference speed**: Real-time predictions require efficient architectures
4. **Explainability**: Consider adding Grad-CAM or SHAP visualizations for clinical trust

The final model selection should balance AUC, F1-score, and computational efficiency based on the experimental results. For screening applications, the model with the highest recall at acceptable precision would be preferred.

---

*Report generated as part of the End-to-End AI System for Pneumonia Detection challenge.*
*Dataset: PneumoniaMNIST (MedMNIST v2) — 28×28 grayscale chest X-rays.*
