# EECS 4414: Network Analysis on Water Contamination

**Project:** Lead Contamination in Toronto’s Water Distribution System  
**Authors:** Saumya Patel, Smith Patel, Krish Patel  
**Institution:** York University, Toronto, ON, Canada  
**Contact:** somy2004@my.yorku.ca, smith04@my.yorku.ca, krish211@my.yorku.ca

---

## Table of Contents
- [Introduction](#introduction)
- [Problem Definition](#problem-definition)
- [Related Work](#related-work)
- [Methodology](#methodology)
  - [Data Sources and Cleaning](#data-sources-and-cleaning)
  - [Network Construction](#network-construction)
  - [Graph Attention Network (GAT) Model](#graph-attention-network-gat-model)
  - [Edge-Level Pipe Analysis](#edge-level-pipe-analysis)
  - [Intervention Scenarios](#intervention-scenarios)
- [Experiments and Evaluation](#experiments-and-evaluation)
  - [Contamination Categorization](#contamination-categorization)
  - [Actual vs Predicted GAT Maps](#actual-vs-predicted-gat-maps)
  - [Material Analysis](#material-analysis)
  - [Intervention Techniques](#intervention-techniques)
  - [ROC/AUC Evaluation](#rocauc-evaluation)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [References](#references)
- [File Structure](#file-structure)
- [Usage](#usage)

---

## Introduction
Lead contamination is a critical concern in municipal water systems worldwide, including Ontario. Aging infrastructure can cause lead to leach into tap water despite treatment.

This project combines multi-year tap-level sampling, watermain infrastructure data, and postal-code coordinates to:
- Categorize lead risk
- Predict contamination using Graph Attention Networks (GAT)
- Simulate intervention strategies to mitigate high-risk areas

---

## Problem Definition
We aim to model and predict lead contamination in Toronto using a postal-area graph:

1. **Contamination Categorization:** Four risk levels (ppb)  
   - Level 0: 0–5  
   - Level 1: 5–10  
   - Level 2: 10–15  
   - Level 3: >15  

2. **GAT-based Prediction:** Predict year-over-year lead levels using node features (3-year mean lead, degree).

3. **Intervention Scenarios:**  
   - High-risk material replacement  
   - Age-targeted renewal  
   - Hotspot remediation  
   - Material-weighted priority replacement  

4. **ROC/AUC Evaluation:** Assess the ability of GAT to distinguish high-risk areas.

---

## Related Work
- Factors like pipe material, age, and historical water chemistry impact lead [Schock & Lytle, 2011].  
- Graph-based time series models improve water quality prediction [Liu et al., 2025].  
- ROC/AUC metrics effectively identify high-risk nodes [Saito & Rehmsmeier, 2015].

---

## Methodology

### Data Sources and Cleaning
- **Lead Samples (2014–2025):** Preprocessing, linear interpolation, conversion to ppb, risk categorization.  
- **Postal-Code Coordinates:** Centroids used as graph nodes; nearest watermain endpoints snapped via KD-tree.  
- **Watermain GeoJSON:** Material, diameter, construction year processed into edges.  

**Panel Construction:** Cartesian product of postal codes × years ensures complete mapping.

---

### Network Construction
- Yearly undirected graphs with postal centroids as nodes and watermain connections as edges.  
- Edge weights based on material, diameter, and age.  
- Node features: 3-year mean lead, node degree (scaled).

---

### Graph Attention Network (GAT) Model
- **Architecture:** 2-layer GAT with multi-head attention (PyTorch Geometric)  
- **Input Features:** 3-year mean lead, node degree  
- **Training:** Preceding 3 years, Adam optimizer, 200 epochs, MSE loss on labeled nodes  
- **Output:** Normalized lead predictions per postal code  

``python
# Example: PyTorch Geometric GAT setup
import torch
from torch_geometric.nn import GATConv

class LeadGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LeadGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*4, out_channels, heads=1, concat=False, dropout=0.6)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).elu()
        x = self.conv2(x, edge_index)
        return x

## Edge-Level Pipe Analysis
- Average GAT predictions of endpoint nodes for each pipe.
- Aggregate pipe attributes: mean age, mean diameter, dominant material.
- Generate CSVs for plotting:


## Intervention Scenarios
1. **High-Risk Material Replacement** – Replace CI, CICL, DIP, DICL, UNK with PVC.  
2. **Age-Targeted Renewal** – Replace pipes older than 50 years.  
3. **Node-Fix Hotspots** – Remediate top 10 high-risk postal codes.  
4. **Material-Weighted Priority** – Replace top 10% risky pipes by combined score.

## Experiments and Evaluation

### Contamination Categorization
- Most samples are Level 0–1; smaller Level 2–3 areas require attention.

### Actual vs Predicted GAT Maps
- Maps for 2020–2026 show model trends vs. measured values.  
- Prediction captures general decline and hotspots but sometimes overestimates reductions.

### Material Analysis
- Material and age correlate with predicted lead risk.  
- AC, CP, PVC show higher predicted leads; age trend weak but noticeable.

### Intervention Techniques
- Simulations show **material-weighted priority replacement** provides the greatest improvement.  
- Hotspot remediation indirectly benefits neighboring nodes.

### ROC/AUC Evaluation
- AUC improved from 0.57 (2024) → 0.63 (2025)  
- MAE ≈ 0.052 ppb, RMSE ≈ 0.072 ppb, R² ≈ 0.8836

## Future Work
- Include schools and daycare datasets (requires geocoding).  
- Extend GAT features for finer infrastructure detail.  
- Explore spatio-temporal GNNs for more accurate predictions.

## Conclusion
The project demonstrates a graph-based framework for analyzing and forecasting lead contamination:
- Risk categorization of samples  
- GAT-based postal-level predictions  
- Pipe-level attribute analysis  
- Intervention simulations  
- ROC/AUC and error evaluation  

This framework supports data-driven decision-making for Toronto’s water infrastructure upgrades.

## References
1. Anaadumba R, Bozkurt Y, et al., *Graph neural network-based water contamination detection*, Front. Environ. Eng., 2025.  
2. Cao J, Zhao D, et al., *Improved Adam optimizer for water quality prediction*, Math. Biosci. Eng., 2023.  
3. Government of Canada, *Lead section*.  
4. Homewater Canada, *Lead Pipes in Homes: A Silent Danger*.  
5. Liu Y, Zheng H, Zhao J, *Enhanced water quality prediction by LSTM and GAT*, Water Research X, 2025.  
6. Schock M.R., Lytle D.A., *Internal corrosion and deposition control*, 2011.  
7. Saito T, Rehmsmeier M., *Precision–recall plot vs ROC*, PLOS ONE, 2015.  
8. University of Toronto, *Toxic lead in Ontario schools and daycares*, 2024.

