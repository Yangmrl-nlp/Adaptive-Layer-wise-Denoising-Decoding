# 🧠 ALD^{2}: Adaptive Layer-wise Denoising Decoding for Large Vision-Language Models

> Mitigating hallucinations in Large Vision-Language Models via layer-wise denoising and multiplicative decoding

📄 **Accepted by Information Processing & Management (一区top，CCF-B)**  

---

## 🚀 Overview

Large Vision-Language Models (LVLMs) suffer from **hallucinations**, often caused by **noisy representations in shallow layers**.

We propose **ALD2**, a decoding-time framework that:

- 🔍 Identifies noisy shallow layers  
- ✂️ Applies visual token pruning  
- ✖️ Uses multiplicative decoding to enhance reliable predictions  

Unlike prior contrastive decoding methods (e.g., VCD, ICD), ALD2:
- Works without modifying model weights  
- Models layer-wise noise explicitly  
- Improves robustness across multiple benchmarks  

---

## 🧩 Method


ALD2 consists of three key components:

### 1. Visual Token Pruning
Removes low-attention visual tokens to suppress noise in shallow layers.

### 2. Adaptive Layer Selection
A lightweight predictor selects the optimal denoising layer dynamically.

### 3. Multiplicative Decoding

Final decoding score:

    ψ(x) = log ( p_final(x) · p_denoised(x) )

This reinforces tokens supported by both shallow and deep layers.

---

## 📊 Results

- 📈 **+5.61%** average improvement on POPE  
- 📈 **+88.56 MME score** on LLaVA-Next-8B  
- 🧠 Significant reduction in hallucinations (CHAIR)  

---

## 🛠️ Setup

### 1. Configure Models & Shots

Edit configuration files:

```
/mnt/data1/yangmrl/ALW_debug/configs
```

You can customize:
- model paths  
- number of shots  
- decoding settings  

---

## 📦 Pipeline

### 2. Construct Training Data

```
bash /path/to/scripts/make_training_data.sh
```

---

### 3. Train Layer Predictor

```
bash /path/to/scripts/train.sh
```

- Backbone: RoBERTa-base  
- Task: layer classification  

---

### 4. Inference

```
bash /path/to/scripts/infer.sh
```

Supported modes:

- `vanilla` → greedy decoding  
- `dola` → DoLa baseline  
- `ald2` → our method  

---

## 🧪 Supported Models

- LLaVA-1.5-7B  
- LLaVA-Next-8B  
- InstructBLIP-Vicuna-7B  
- InternVL-3.5-8B  

---

## ⚖️ Trade-offs

| Aspect        | ALD2 |
|--------------|------|
| Accuracy      | ⭐⭐⭐⭐ |
| Hallucination | 🔻 Reduced |
| Latency       | ⬆ Higher than greedy |

---

## 📌 Key Insights

- Hallucinations originate from shallow-layer noise  
- Early layers contain useful but noisy signals  
- Denoising + multiplicative fusion improves reliability  

---

## 🔓 Release

> Code and data will be released after publication.

---

## 📖 Citation

```
@article{ald2,
  title={Adaptive Layer-wise Denoising Decoding for Hallucinations Mitigation in Large Vision-Language Models},
  journal={Information Processing & Management},
  year={2026}
}
```

---

## ⭐ Acknowledgements

Inspired by:
- Contrastive Decoding (VCD, ICD)  
- DoLa  
- Layer-wise analysis in LLMs  
