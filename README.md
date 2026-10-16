# ml-capstone

| Track | Notebook | Owner |
|---|---|---|
| Regression | `notebooks/regression.ipynb` | [@arhamgarg](https://github.com/arhamgarg) |
| Classification, Part A | `notebooks/classification.ipynb` | [@fingernailz](https://github.com/fingernailz) |
| Classification, Part B | `notebooks/classification.ipynb` | [@arhamgarg](https://github.com/arhamgarg) |
| Clustering | `notebooks/clustering.ipynb` | [@fingernailz](https://github.com/fingernailz) |

## Dataset descriptions and problem statements

| Track | Dataset | Size | Problem |
|---|---|---|---|
| Regression | [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294) | 9,568 × 5 | Predict a power plant's net hourly output (`PE`, MW) from temperature, exhaust vacuum, pressure and humidity. |
| Classification | [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350) | 30,000 × 24 | Predict whether a client defaults on next month's payment from their credit limit, demographics and six months of bills and repayments. 22.1% of clients default. |
| Clustering | [Wholesale Customers](https://archive.ics.uci.edu/dataset/292) | 440 × 8 | Group customers by annual spending in six product categories. |

## Results

### Regression

| Model | R² | RMSE (MW) | MAE (MW) |
|---|---|---|---|
| **Random Forest** | **0.9611** | **3.365** | **2.424** |
| K-Nearest Neighbors | 0.9558 | 3.586 | 2.626 |
| Gradient Boosting | 0.9543 | 3.646 | 2.767 |
| Support Vector Regressor | 0.9475 | 3.908 | 2.958 |
| Polynomial Regression | 0.9446 | 4.012 | 3.116 |
| Decision Tree | 0.9441 | 4.033 | 2.961 |
| ElasticNet | 0.9363 | 4.304 | 3.411 |
| Lasso | 0.9363 | 4.306 | 3.407 |
| Ridge | 0.9362 | 4.307 | 3.407 |
| Linear Regression | 0.9362 | 4.307 | 3.407 |

Random Forest is our final model. Its five-fold CV R² is 0.9581 ± 0.0063, and its RMSE of 3.4 MW is under 1% of the average output. The linear models all stall around 0.936, so the relationship isn't linear. The `AT_x_V` (temperature × vacuum) feature we added gets 0.90 of Random Forest's feature importance.

### Classification

| Model | Part | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | B | **0.817** | **0.800** | **0.817** | **0.797** | **0.774** |
| Random Forest | B | 0.817 | 0.799 | 0.817 | 0.796 | 0.773 |
| Decision Tree | A | 0.815 | 0.796 | 0.815 | 0.796 | 0.757 |
| Bagging | B | 0.817 | 0.799 | 0.817 | 0.795 | 0.773 |
| MLP | B | 0.816 | 0.798 | 0.816 | 0.794 | 0.763 |
| SVM | A | 0.816 | 0.798 | 0.816 | 0.793 | 0.712 |
| AdaBoost | B | 0.817 | 0.800 | 0.817 | 0.791 | 0.774 |
| K-Nearest Neighbors | A | 0.809 | 0.787 | 0.809 | 0.786 | 0.738 |
| Logistic Regression | A | 0.812 | 0.792 | 0.812 | 0.781 | 0.750 |
| Naive Bayes | A | 0.389 | 0.730 | 0.389 | 0.391 | 0.680 |

Gradient Boosting is our final model, though the top seven are within 0.006 F1 of each other and Random Forest is basically tied in cross-validation (0.7998 vs 0.7995). Tuning took its F1 from 0.7949 to 0.7967. The bigger problem is that it only catches 483 of the 1,326 defaulters. The 82% accuracy mostly reflects the 78% of clients who don't default. The latest repayment status (`PAY_0`) and our `DELAY_MONTHS_COUNT` feature carry most of the signal.

### Clustering

All models use k = 2, picked from the elbow curve and silhouette scores.

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Cluster sizes |
|---|---|---|---|---|
| **K-Means** | 0.290 | 1.352 | **189.050** | 252, 188 |
| Agglomerative (Ward) | 0.258 | 1.600 | 134.624 | 178, 262 |
| Agglomerative (complete) | 0.322 | 1.554 | 61.992 | 38, 402 |
| Agglomerative (average) | 0.569 | 0.883 | 13.965 | 2, 438 |
| Agglomerative (single) | 0.570 | 0.307 | 8.964 | 439, 1 |

K-Means is our final model. Average and single linkage look best on silhouette and Davies-Bouldin, but they just split off one or two outliers. K-Means gives two real groups and stays stable when refit on 80% subsamples (silhouette std 0.004). The groups line up with business type. One buys more fresh and frozen food and is 96.8% hotels, restaurants and cafés. The other spends about 5× more on groceries and 13× more on detergents and paper, and is 71.3% retailers.

## Environment setup

```bash
git clone https://github.com/arhamgarg/ml-capstone.git
cd ml-capstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
python data/download.py
jupyter lab notebooks/
streamlit run app/app.py
```

## AI use

Generative AI was used for code scaffolding and this README. All analysis, interpretation and feature-engineering decisions in the notebooks are the team's own.

## License

[AGPL-3.0](LICENSE)
