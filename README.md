# Optimized-CSP-Pipeline-for-BCI-Competition-IV-2a-Motor-Imagery-Classification.

## Project Summary
This repository contains the complete implementation and scientific report for an optimized Brain-Computer Interface (BCI) pipeline designed for the binary classification of **Left Hand vs. Right Hand** motor imagery (MI).

The pipeline integrates advanced signal processing techniques with rigorous machine learning optimization, achieving a peak accuracy of **84.5%** on the BCI Competition IV 2a dataset (Subject A01).

## Methodology Highlights
* **Preprocessing:** Current Source Density (CSD) transformation was applied to enhance spatial resolution of EEG signals.
* **Filtering:** FIR Bandpass Filtering (13.0–28.0 Hz) was optimized for the Beta rhythm (ERD/ERS) band, critical for MI feature detection.
* **Feature Extraction:** Common Spatial Patterns (CSP) were used to extract the most discriminative features, using 4 optimal components.
* **Classification:** The features were classified using a Support Vector Machine (SVC) classifier.
* **Results:** The final cross-validation accuracy of **84.5% ± 6.2%** is reported in the final scientific paper.

## Repository Structure
The project files are organized for clarity and full reproducibility:
