# Benchmarking Architectures for Churn Prediction: The Impact of Data Leakage

**Authors:** Neta Ben Mordechai, Itay Chabra, Yuval Gefen

This repository contains the official implementation and experimental data for the study **"Benchmarking Architectures for Churn Prediction: The Impact of Data Leakage"**.

## Project Overview
Customer Churn Prediction is a critical task in the industry. Recent studies (e.g., *ChurnNet*) have claimed near-perfect accuracy using 1D-CNNs and aggressive oversampling. Our research demonstrates that these results are often artifacts of **Data Leakage** caused by improper preprocessing pipelines (specifically, applying SMOTE before cross-validation splitting).

This project implements a **Strict, Leakage-Free Pipeline** to truthfully benchmark modern architectures:
1.  **Tree-Based Models:** XGBoost, LightGBM (Industry Standards).
2.  **Deep Learning for Tabular Data:** FT-Transformer, TabNet, Attentive MLP.
3.  **Control Model:** 1D-CNN (Reproduction of ChurnNet).

## Key Findings
* **Data Leakage Matter:** "State-of-the-art" results dropped significantly when the validation pipeline was corrected.
* **Trees are King:** XGBoost consistently outperformed complex Deep Learning models in stability and training efficiency on tabular data.
* **Transformers are Close:** The FT-Transformer was the only Deep Learning model to compete with XGBoost, offering superior Recall in high-signal environments.
* **Data Quality > Model Complexity:** All models struggled on the Banking dataset (F1 ~0.62) compared to the Telecom dataset (F1 ~0.85) due to the lack of behavioral features.

## Repository Structure
* `Notebooks/`: Contains Jupyter notebooks for all experiments detailed in the paper.
* `Datasets/`: The links to the datasets (Telecom UCI & Bank Customer Churn).
* `Article`: The full research paper (PDF).
