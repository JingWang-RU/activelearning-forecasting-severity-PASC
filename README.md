# Activ­e Learn­ing for Forec­astin­g Sever­ity among Patie­nts with Post Acute Seque­lae of SARS-CoV-2   
[![PyPI version](https://badge.fury.io/py/python.svg)](https://pypi.org/project/python/)

The **Active Learning Clinical Risk Forecasting** library provides a framework for predicting clinical risk using time series of events extracted from case reports. It is built with Python 3.8+, SentenceTransformer(`neuml/pubmedbert-base-embeddings`), optional GPU for modeling, including feature importance through attention-based mechanisms. The workflow supports active learning via both uncertainty sampling and random sampling strategies. The dataset is also available on huggingface "juliawang2024/longcovid-risk-eventtimeseries"

The codebase includes:

- `ac_forecasting_321.py` – the main program containing the model and training pipeline.
- `ac_end_end.ipynb` – an end-to-end notebook to load data, train the model, and evaluate results.

---

## 📂 Data

- **`am_risk_annote-310.csv`**  
  This file contains risk annotations (`0` or `1`) for each case report.

- **Text time series per case report**  
  Stored in the directory `am_18_llm`, where each `.txt` file contains a sequence of clinical events and associated timestamps.

---

## 📄 Documentation

- The data loading, training, and testing process is demonstrated in [`ac_end_end.ipynb`](./ac_end_end.ipynb).
- The model code and utilities are located in [`ac_forecasting_321.py`](./ac_forecasting_321.py).

## 📄 Citation
title={Active Learning for Forecasting Severity among Patients with Post Acute Sequelae of SARS-CoV-2},
author={Wang, Jing and and Sra, Amar and Weiss, C. Jeremy},
journal={AMIA},
year={2025}
}