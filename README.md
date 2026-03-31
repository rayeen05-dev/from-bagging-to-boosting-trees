#  From Bagging to Boosting Trees — Experiments & Insights

This repository is a collection of hands-on experiments conducted while learning core Machine Learning concepts, specifically **Decision Trees and Ensemble Methods**.

It is **not a production project**, but a structured exploration of how models behave under different conditions.

---

# 🎯 Objective

The goal of this work was to:

- Understand Decision Trees beyond theory
- Explore ensemble methods (Bagging, Random Forest, AdaBoost)
- Experiment with model behavior on noisy, non-linear data
- Learn from mistakes and unexpected results

---

# 📊 Dataset

```python
x, y = make_moons(n_samples=10000, noise=0.4)
```

### Why `make_moons`?

- Non-linear dataset → requires flexible models  
- Noise = 0.4 → introduces real-world difficulty  
- Helps visualize decision boundaries  

### Train/Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
```

---

# 🌳 Step 1 — Decision Tree Baseline

```python
tree = DecisionTreeClassifier(
    random_state=42,
    ccp_alpha=0.0005,
    criterion="gini",
    max_depth=1,
    max_leaf_nodes=4,
    min_samples_leaf=50
)
```

### Observations

- Very shallow tree → **underfitting**
- Strong constraints limit learning capacity

### Takeaways

- Decision Trees are highly sensitive to hyperparameters
- Too simple → underfit  
- Too complex → overfit  

---

# 🧪 Step 2 — Manual Bagging (From Scratch)

### What I did

- Created subsets using `ShuffleSplit`
- Trained multiple Decision Trees
- Combined predictions using **majority voting**

```python
ss = ShuffleSplit(n_splits=100, train_size=2000, random_state=42)
```

```python
def predict_ensemble(models, x):
    predictions = np.array([m.predict(x) for m in models])
    return mode(predictions, axis=0).mode[0]
```

### Problem

❗ The ensemble performed poorly and was **not comparable to Random Forest**

### Why?

- Each model saw only a small subset of data  
- Trees were too weak  
- No feature randomness  
- Models were highly correlated  

### Key Insight

> More models ≠ better performance

---

# 🌲 Step 3 — Random Forest

### What I did

- Used `RandomForestClassifier`
- Applied `GridSearchCV` for tuning

```python
param_grid = {
    "n_estimators": [100, 200],
    "criterion": ['gini', 'entropy'],
    "max_depth": [10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt"]
}
```

### Why it worked better

- Bootstrap sampling  
- Feature randomness  
- Stronger trees  

### Takeaways

- Random Forest reduces correlation between trees  
- Feature randomness is critical  
- Proper ensembles outperform naive implementations  

---

# 📦 Step 4 — BaggingClassifier (Sklearn)

```python
bag_clf = BaggingClassifier(
    tree,
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    max_samples=0.8,
    max_features=0.8
)
```

### Observations

- Easier and more reliable than manual implementation  
- Better performance due to correct sampling  

---

# 🚀 Step 5 — AdaBoost

```python
ada = AdaBoostClassifier(
    estimator=tree,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
```

### Observations

- Sequential learning (focus on mistakes)  
- Performance depends on base model quality  

### Takeaways

- Boosting ≠ Bagging  
- Weak learners must still be meaningful  
- Overly simple trees limit performance  

---

# ⚠️ Challenges & Mistakes

- Built an ensemble with weak models → poor results  
- Assumed more models = better performance  
- Ignored feature randomness initially  
- Over-constrained Decision Trees  

---

# 🧠 Final Conclusions

1. Model quality > number of models  
2. Random Forest > naive bagging  
3. Feature randomness is essential  
4. Boosting and Bagging solve different problems  
5. Implementing manually reveals hidden complexity  

---

# 📌 What This Repo Represents

This project reflects:

- Learning through experimentation  
- Understanding through failure  
- Building intuition, not just using libraries  

---


# ⚠️ Disclaimer

This is a **learning-focused repository**.  
Code is not optimized or production-ready, and that is intentional.
