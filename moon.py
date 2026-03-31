from sklearn.datasets import make_moons 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import mode
from sklearn.ensemble import BaggingClassifier 
x,y = make_moons(n_samples=10000, noise =0.4)
"""print(x)
print(y)"""

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)



"""plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=10)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Set (3000 samples)")
plt.show()"""
"""param_grid = {
    "max_depth" : [5,10,15,20,23],
    "min_samples_leaf" : [50,100,150,200],
  
    "criterion" : ["gini" , "entropy","log_loss"],
    "ccp_alpha": [0.0005, 0.001, 0.005]
}"""

#MANUAL BAGGING ENSEMBLE 
"""models = []
#cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
ss = ShuffleSplit(n_splits=100,train_size=2000,test_size=None,random_state=42)
for train_index , _ in ss.split(x_train) : 
    x_subset = x_train[train_index]
    y_subset = y_train[train_index]
    tree = DecisionTreeClassifier(random_state=42,ccp_alpha=0.0005,criterion="gini",max_depth=5,max_leaf_nodes=4,min_samples_leaf=50)
    tree.fit(x_subset,y_subset)
    models.append(tree)

print(len(models))"""
tree = DecisionTreeClassifier(random_state=42,ccp_alpha=0.0005,criterion="gini",max_depth=1,max_leaf_nodes=4,min_samples_leaf=50)
"""print("best param : " , grid.best_params_)
print("with a score : " , grid.best_score_)
best_model = grid.best_estimator_
test_score = best_model.score(x_test, y_test)

print("Test score:", test_score)"""

#COMBINING ALL TREES TOGETHER 
"""def predict_ensemble(models , x ) : 
    predictions = np.array([m.predict(x) for m in models ])
    return mode(predictions , axis=0).mode[0]"""

#testing it 
"""y_pred = predict_ensemble(models,x_test)
accuracy = np.mean(y_pred == y_test)
print("ensemble accuracy :" , accuracy)"""
#=> not equivalent to a randomforest (very low accuracy )
#each model is seeing just a bit of the trained data
#more models doesnt mean good ones 

#RANDOMFOREST CLASSIFIER 
"""cv=StratifiedKFold(n_splits=5,shuffle=True ,random_state=42)
param_grid = {
    "n_estimators" : [100,200],
    "criterion" : ['gini','entropy'],
    "max_depth" : [10,20],
    "min_samples_leaf" : [1,5,10],
    "max_features" : ["sqrt"]
    
   
}
forest = RandomForestClassifier(random_state=42,n_jobs=5)
grid = GridSearchCV(forest,param_grid,scoring="accuracy",cv=cv)
grid.fit(x_train,y_train)

print("Best params:", grid.best_params_)
print("CV score:", grid.best_score_)

best_model = grid.best_estimator_
print("Test score:", best_model.score(x_test, y_test))"""


"""bag_clf = BaggingClassifier(
    tree,n_estimators=500,bootstrap=True,n_jobs=-1,
    max_samples = 0.8,
    max_features=0.8
)
bag_clf.fit(x_train,y_train)
print("model_score:" , bag_clf.score(x_test,y_test))"""

# trying adaboost 
ada = AdaBoostClassifier(estimator=tree,n_estimators=100,learning_rate=0.5,random_state=42)
ada.fit(x_train,y_train)
print("adaboost accuraccy : ",ada.score(x_test,y_test))