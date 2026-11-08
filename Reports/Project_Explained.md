# Project Explained – End-to-End Guide (for viva preparation)

> Simple English with a little Hindi. Read this top to bottom once, then use the "Numbers to remember"
> and "Questions" sections before the viva.

---

## 1. The project in 60 seconds

**Problem:** Roads get jammed at certain hours. If a traffic department knows *in advance* how much traffic
will come, it can change signal timings, divert traffic and send police **before** the jam happens.

**What I built:** A machine-learning system that looks at **6 years of past hourly traffic data** from the
I-94 highway in the USA and learns patterns. It answers two questions:

1. **How many vehicles will pass in a given hour?** → a number → **Regression** (Module 1)
2. **Will that hour be Low, Medium or High congestion?** → a category → **Classification** (Module 2)

Then I put both models in a **Streamlit website (dashboard)** where anyone can choose a date, hour and
weather and see the prediction.

> Hindi mein: "Purane traffic data se machine ko sikhaya ki kis time kitni gaadiyan aati hain, aur us hisaab
> se future ka traffic aur jam (Low/Medium/High) predict karta hai."

**Best result:** Random Forest – R² = **0.96** for volume, **91%** accuracy (macro F1 0.913) for congestion.

---

## 2. Architecture – the complete flow

```
 ┌──────────────────────────┐
 │ 1. RAW DATA (UCI)         │  Dataset/Metro_Interstate_Traffic_Volume.csv.gz
 │ 48,204 rows × 9 columns   │  (hourly traffic + weather + holiday, 2012–2018)
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 2. CLEANING               │  src/data_preprocessing.py → clean_data()
 │ duplicates, outliers,     │  Notebook 01
 │ datatypes, holiday fix    │  → Dataset/traffic_clean.csv (40,564 rows)
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 3. FEATURE ENGINEERING    │  src/data_preprocessing.py → add_features()
 │ hour, weekday, weekend,   │  + congestion label (Low/Medium/High)
 │ rush hour, sin/cos, etc.  │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4. EDA + STATISTICS       │  Notebook 02 – charts, correlation,
 │ "What affects traffic?"   │  t-test, ANOVA, chi-square
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 5. TIME-BASED SPLIT       │  src/model_utils.py → time_split()
 │ Train: Oct 2012–Sep 2017  │
 │ Test : Oct 2017–Sep 2018  │  (future year the model never saw)
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 6. PIPELINE               │  src/model_utils.py → build_preprocessor()
 │ Scale numbers + One-hot   │  Pipeline = preprocessing + model together
 │ categories + model        │  (prevents data leakage)
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐      ┌──────────────────────────┐
 │ 7a. REGRESSION (Nb 03)    │      │ 7b. CLASSIFICATION (Nb 04)│
 │ 8 models, GridSearchCV    │      │ 7 models, GridSearchCV    │
 │ Best: Random Forest       │      │ Best: Random Forest       │
 │ → models/regressor.joblib │      │ → models/classifier.joblib│
 └────────────┬─────────────┘      └────────────┬─────────────┘
              └───────────────┬──────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────┐
 │ 8. STREAMLIT DASHBOARD   app/app.py               │
 │ user inputs → same features → both models →       │
 │ volume + congestion + 24-hour chart + suggestion  │
 │ Deployed on Streamlit Cloud (public link)         │
 └──────────────────────────────────────────────────┘
```

---

## 3. Which file does what (folder map)

| File / folder | What it does (simple words) |
|---|---|
| `Dataset/Metro_Interstate_Traffic_Volume.csv.gz` | Original data downloaded from UCI (compressed) |
| `Dataset/traffic_clean.csv` | Data after cleaning |
| `src/data_preprocessing.py` | **Heart of data work**: loading, cleaning, new features, congestion label |
| `src/model_utils.py` | Time-based split, preprocessing (scaling + one-hot), cross-validation setting |
| `src/train_models.py` | Re-trains the final 2 models using the best settings (used by the dashboard) |
| `Notebooks/01_data_cleaning.ipynb` | Shows the cleaning step by step with plots |
| `Notebooks/02_eda.ipynb` | Charts + statistical tests + interpretation |
| `Notebooks/03_regression.ipynb` | Module 1 – 8 regression models compared |
| `Notebooks/04_classification.ipynb` | Module 2 – 7 classification models compared |
| `models/best_params_*.json` | Best hyperparameters found by GridSearchCV |
| `app/app.py` | The Streamlit dashboard |
| `app/requirements.txt` | Libraries needed by the online dashboard |
| `Images/` | All charts (used in report and README) |
| `Reports/` | Report (.docx), result tables (.csv), this guide |
| `requirements.txt` | All libraries for the full project |

**Why `src/` files + notebooks?** Cleaning code is written **once** in `src/` and *imported* by the notebooks
and the dashboard. So the dashboard prepares input **exactly the same way** as training. (Same recipe
everywhere – no mismatch.)

---

## 4. Step-by-step: what I did and WHY

### Step 1 – Dataset
* UCI Metro Interstate Traffic Volume: hourly data from a highway between Minneapolis and St Paul, 2012–2018.
* Columns: holiday, temp (Kelvin), rain_1h, snow_1h, clouds_all, weather_main, weather_description,
  date_time, **traffic_volume (target)**.
* **Why this dataset?** Public and citable, large (6 years), has exactly what a traffic department knows in
  advance (date, hour, holiday, weather), and the target is a number – so it works for regression *and*,
  by dividing into ranges, for classification.

### Step 2 – Cleaning (Notebook 01)
| Problem found | What I did | Why |
|---|---|---|
| `date_time` was text | converted to datetime | so I can extract hour, day, month |
| 17 exact duplicate rows | removed | same row counted twice = wrong learning |
| 7,612 repeated hours (one row per weather type, e.g. Rain + Mist in same hour) | kept one row per hour | the traffic number was the same; it would double count that hour |
| 10 rows with temp = 0 Kelvin (−273 °C) | removed | impossible – sensor error |
| 1 row with 9,831 mm rain in 1 hour | removed | impossible – sensor error |
| Holiday name only on the 12 AM row | spread to all 24 hours of that day (`is_holiday`) | whole day is a holiday, not just midnight |
| Temperature in Kelvin | converted to °C | easier to understand |
| Missing values | none really (“None” in holiday just means no holiday) | – |
| Gap in 2014–2015 | **not** filled | inventing traffic numbers would be guessing; still ~40k rows left |

Result: **48,204 → 40,564 rows**.

### Step 3 – Feature engineering (making new useful columns)
| Feature | Meaning | Why useful |
|---|---|---|
| `hour` | 0–23 | traffic depends strongly on time |
| `day_of_week` | 0 = Mon … 6 = Sun | weekdays busier |
| `month` | 1–12 | seasonal effect (small) |
| `is_weekend` | 1 if Sat/Sun | weekend pattern different |
| `is_holiday` | 1 if holiday | holidays are quieter |
| `is_rush_hour` | 1 if 7–9 AM or 4–6 PM on a working day | captures office traffic |
| `hour_sin`, `hour_cos` | hour placed on a circle | 11 PM and 12 AM are neighbours; plain numbers 23 and 0 look far apart |
| `daytype_hour` | e.g. `work_8`, `off_8` | 8 AM on Monday ≠ 8 AM on Sunday; lets even linear models learn both shapes |
| `temp_c` | temperature in °C | – |
| `congestion_level` | Low < 2000, Medium 2000–4499, High ≥ 4500 | **target for Module 2** |

**Why thresholds 2000 and 4500?** They split the data into roughly equal thirds (31% / 33% / 36%), so the
classes are balanced, and they are round numbers that are easy to explain.

### Step 4 – EDA (Notebook 02) – "what affects traffic?"
* **Univariate** (one column): traffic is **bimodal** – one peak at night (low) and one in daytime (high).
  Rain and snow are 0 most of the time.
* **Bivariate** (traffic vs one column): weekdays have **two peaks – 7 AM and 4–5 PM** (office going/returning).
  Weekends have one gentle midday hump.
* **Multivariate**: the **day × hour heatmap** – the most important picture of the project. Red = jam.
* **Correlation**: `hour_cos` (−0.77) and `is_rush_hour` (0.52) are strongest; weather ≈ 0.
* **Statistical tests** (p < 0.05 means "difference is real, not chance"):
  * **t-test**: weekday daytime 5,074 vs weekend 3,703 vehicles/hour → significant (≈37% more).
  * **ANOVA**: weather types differ significantly, but only ~10% → weather is a **weak** factor.
  * **Chi-square**: congestion level depends on weekday/weekend (45% High on weekdays vs 12% on weekends),
    Cramér's V = 0.31 = moderate.

**Conclusion of EDA:** Time (hour + weekday/holiday) is the main driver; weather is minor; the relation is
non-linear → tree models will likely win.

### Step 5 – Train/test split (time-based)
* Train = Oct 2012 – Sep 2017, Test = Oct 2017 – Sep 2018.
* **Why not random split?** In real life we predict the **future**. A random split would put, say, 8 AM in the
  test set and 7 AM and 9 AM of the same day in training – the model would "cheat" by peeking at neighbours.
  Testing on a future year is the honest check.

### Step 6 – Pipeline and data leakage
* **Data leakage** = information from the test data sneaking into training (makes results look better than
  reality).
* I used a scikit-learn **Pipeline**: `preprocessing → model`. The scaler/encoder learn only from the
  training data inside each fold.
* Preprocessing:
  * **StandardScaler** for numbers (mean 0, std 1) – needed by Linear/Logistic Regression, KNN, SVM (they
    are sensitive to scale). Trees don't need it but it doesn't hurt.
  * **OneHotEncoder** for categories – e.g. `hour` becomes 24 yes/no columns, so the model can learn a
    separate effect for each hour.
* For classification, `traffic_volume` is **not** used as an input (the label is made from it → that would
  be cheating).

### Step 7 – Hyperparameter tuning
* **Hyperparameter** = a setting we choose before training (e.g. tree depth, k in KNN, alpha in Ridge).
* **GridSearchCV** tries every combination and keeps the best.
* **TimeSeriesSplit (3 folds)** – each fold trains on earlier data and validates on the next period
  (like a mini future test).

### Step 8 – Models and results

**Module 1 – Regression (predict the number)**

| Model | Idea in one line | Test R² |
|---|---|---|
| Baseline | always predicts the average | 0.000 |
| Linear Regression | straight-line relationship | 0.946 |
| Polynomial Regression | adds curved terms (x², x³) | 0.948 |
| Ridge | linear + L2 penalty (shrinks coefficients) | 0.946 |
| Lasso | linear + L1 penalty (can make coefficients 0 → feature selection) | 0.946 |
| Elastic Net | mix of L1 and L2 | 0.945 |
| Decision Tree | if-else rules | 0.955 |
| **Random Forest** | **150 trees, average their answers** | **0.960** |

Metrics: **MAE** = average error in vehicles (235); **RMSE** = like MAE but punishes big errors more (395);
**R²** = how much of the variation is explained (0.96 = 96%).

**Module 2 – Classification (predict Low/Medium/High)**

| Model | Idea | Macro F1 |
|---|---|---|
| Baseline | always says the most common class (High) | 0.180 |
| Logistic Regression | linear boundary + probabilities | 0.903 |
| KNN (k = 15) | looks at the 15 most similar hours | 0.909 |
| Naive Bayes | Bayes theorem, assumes features independent | 0.852 |
| Decision Tree | if-else rules | 0.911 |
| **Random Forest** | **many trees vote** | **0.913** |
| SVM (RBF) | widest-margin boundary in a curved space | 0.907 |

Metrics: **Accuracy** (correct %), **Precision** (when model says High, how often is it right),
**Recall** (of real High hours, how many did it catch), **F1** (balance of precision and recall),
**ROC-AUC** (how well it ranks; 0.5 = random, 1 = perfect), **Confusion matrix** (table of right/wrong).

**Why Random Forest wins:** traffic depends on combinations (hour × weekday × weather); trees capture such
non-linear interactions; averaging many trees reduces overfitting. Train R² 0.969 vs test 0.960 → small gap
→ no serious overfitting.

### Step 9 – Interpretation (what the model learned)
* Most important features: `hour_cos`, `hour_sin`, `is_weekend`, `daytype_hour`, `is_holiday`. Weather
  barely matters – same as EDA said.
* Errors are biggest in rush hours (accidents/events change traffic a lot) and smallest at night.
* **Blizzard example:** 14–15 April 2018 – model predicted normal weekend traffic but real traffic was low
  because of a record snowstorm. Snow is badly recorded in the dataset → model couldn't know.
  ("Model utna hi accha hai jitna data.")
* Confusion matrix: mistakes are only between **neighbour** classes; **0** High hours predicted as Low.

### Step 10 – Dashboard and deployment
* `app/app.py` (Streamlit): sidebar inputs → `build_input()` creates the **same features** using
  `add_features()` → regressor predicts volume, classifier predicts level → shows result, suggestion,
  24-hour colour chart, best hours to travel.
* Model files (`.joblib`) are large, so they are **not** on GitHub. On the first start, the app runs
  `train_models.py` to train them using the best settings saved in `models/*.json`, then caches them.
* Deployed on **Streamlit Community Cloud**:
  https://anwarhusain150-smart-traffic-management-system-appapp-h521pm.streamlit.app/

### Step 11 – Git and GitHub
* Each feature was built on its **own branch** (`feature/data-cleaning`, `feature/eda`, …) and merged into
  `main` through a **Pull Request** – the professional way teams work.

---

## 5. Numbers to remember

| What | Value |
|---|---|
| Raw rows / clean rows | 48,204 / 40,564 |
| Train / test hours | 31,831 / 8,733 |
| Class split | Low 30.9% · Medium 33.3% · High 35.8% |
| Weekday vs weekend daytime mean | 5,074 vs 3,703 |
| Best regression | Random Forest – MAE 235, RMSE 395, R² 0.960 |
| Best linear model | R² 0.946 |
| Best classification | Random Forest – Acc 0.912, F1 0.913, ROC-AUC 0.982, High recall 0.919 |
| Peak hours | 7 AM and 4–5 PM on weekdays |

---

## 6. Questions the professor may ask (short answers)

1. **Why is this regression AND classification?** Volume is a number → regression. Congestion level is a
   category → classification. Same data, two useful outputs.
2. **Why time-based split, not random?** We predict the future; random split leaks neighbouring hours and
   gives over-optimistic results.
3. **What is data leakage and how did you avoid it?** Test information entering training. Avoided with a
   Pipeline (scaler fitted only on training folds), time split, and not using `traffic_volume` as a classifier input.
4. **Why one-hot encode hour instead of using it as a number?** The effect of hour is not a straight line
   (goes up at 7 AM, down at 10 AM, up again at 4 PM). One-hot lets the model learn each hour separately.
5. **Why sin/cos of hour?** To show the cycle: 23:00 and 00:00 are close in time.
6. **Why scaling?** Distance/gradient models (KNN, SVM, Logistic, Linear with regularisation) are affected
   by feature size; scaling puts all features on the same scale.
7. **What is overfitting? Is your model overfitting?** Learning noise of training data. Train R² 0.969 vs
   test 0.960 → very small gap → no.
8. **Why Random Forest is best?** Captures non-linear interactions, averaging many trees lowers variance;
   it had the best CV and test scores.
9. **Ridge vs Lasso?** Ridge (L2) shrinks coefficients; Lasso (L1) can make some exactly 0 – it zeroed 33 of
   110 columns.
10. **Why macro F1 and not only accuracy?** Accuracy can hide poor performance on one class; macro F1
    treats all classes equally. Baseline shows this: High recall = 1.0 but F1 = 0.18.
11. **What does ROC-AUC 0.98 mean?** 98% chance the model ranks a random positive hour above a random
    negative hour – excellent separation.
12. **Why Naive Bayes is weakest?** It assumes features are independent; ours are related (hour, hour_sin,
    is_rush_hour).
13. **Why was SVM trained on a subset?** SVM training time grows ~ n²; used every 3rd row (~10,600).
14. **What does the p-value mean?** Probability of seeing such a difference by chance if there was no real
    difference. < 0.05 → reject the null hypothesis.
15. **Limitations?** No accident/event data, poor snow data, one road only, thresholds are my choice,
    2014–15 data gap.
16. **Improvements?** Lag features (traffic of previous hours), event data, gradient boosting, live data,
    prediction intervals.
17. **How does the dashboard make a prediction?** Inputs → same feature function as training → saved
    pipeline (preprocess + Random Forest) → prediction.

---

## 7. How to demo in the viva (2 minutes)
1. Open the live dashboard → set **Monday, 8 AM** (Clouds, 15 °C) → ~5,800 vehicles/hour, **High**.
   Change to **3 AM** → ~400, **Low**.
2. Tick **Public holiday** at 8 AM → volume drops to ~4,200.
   *Note:* the classifier may still show **High** although 4,200 is in the Medium range. The regressor and the
   classifier are **two separate models**; near a threshold they can disagree. If asked: "Both are estimates;
   4,200 is close to the 4,500 boundary, so either label is reasonable. A future improvement is to derive the
   class from the regression output (which in my test gave F1 0.916) to keep them consistent."
3. Show the 24-hour chart – red bars at rush hours.
4. Open GitHub → README → **Pull requests → Closed** (8 PRs) → Commits history.
5. Open `Notebooks/02_eda.ipynb` → heatmap; `04_classification.ipynb` → confusion matrix.
