# DeepFakeDetection

Repository for deep fake detection algorithm.

## Architecture

High-level pipeline and model flow (HSF-CVIT–style design: modular data pipeline, pluggable model block, training/validation, and evaluation).

```mermaid
flowchart TB
 subgraph Data_Pipeline["1. Data Pipeline - Modular"]
    direction TB
        DP1["Data Preprocessing"]
        D1[("Datasets:<br>FaceForensics++ & Celeb-DF")]
        DP2["Identity-Level Split"]
        DP3["Data Augmentation:<br>Crop, Flip, Jitter, Noise"]
        DP4["Batch Generation"]
  end
 subgraph HSF_CVIT["Proposed HSF-CVIT Model"]
    direction TB
        Branch1["Spatial Branch:<br>EfficientNet-B4"]
        Branch2["Frequency Branch:<br>SRM Filters + DCT"]
        Fusion["Cross-Attention ViT Fusion"]
  end
 subgraph Model_Block["2. Model Architecture - Pluggable Interface"]
    direction TB
        M_Input["Input Image Tensors"]
        HSF_CVIT
        M_Output["Prediction Logits"]
  end
 subgraph Training_Validation["3. Training & Validation Engine"]
    direction TB
        TR1["Mixed-Precision Training Loop"]
        TR2["Adam Optimizer"]
        TR3["Loss Calculation"]
        TR4["Regularization:<br>Dropout & Weight Decay"]
  end
 subgraph Evaluation["4. Evaluation & Metrics"]
    direction TB
        E1["ROC-AUC"]
        E2["Precision & Recall"]
        E3["F1 Score"]
  end
    D1 --> DP1
    DP1 --> DP2
    DP2 --> DP3
    DP3 --> DP4
    Branch1 --> Fusion
    Branch2 --> Fusion
    M_Input --> HSF_CVIT
    HSF_CVIT --> M_Output
    TR1 --- TR2 & TR3 & TR4
    DP4 -- Train/Val Data Dataloaders --> M_Input
    M_Output -- Forward Pass --> Training_Validation
    Training_Validation -- Validation Phase --> Evaluation

     HSF_CVIT:::pluggable
    classDef pluggable fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,stroke-dasharray: 5 5
```
