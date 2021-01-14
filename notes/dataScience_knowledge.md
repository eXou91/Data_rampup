# **Cross Validation**

La validation croisée (ou cross-validation  en anglais) est une méthode statistique qui permet d'évaluer la capacité de généralisation d'un modèle. Il s'agit d'une méthode qui est plus stable et fiable que celle d'évaluer la performance sur des données réservées pour cette tache (Hold-out Validation). Généralement lorsqu'on parle de cross-validation (cv), l'on réfère à sa variante la plus populaire qu'est le k-fold cross-validation. Dans ce cas, nous profitons de toutes les données à disposition en les divisant en k parties égales (folds) sur lesquelles on entraîne et teste un modèle pendant k itérations. A chaque itération, le modèle est entrainé sur k-1 folds et est testé sur le fold restant

https://www.kevindegila.com/la-validation-croisee-en-machine-learning-tout-ce-quil-faut-savoir/



# **Bias–variance tradeoff** : 


## **Error due to Bias**: 
The error due to bias is taken as the difference between the expected (or average) prediction of our model and the correct value which we are trying to predict. Of course you only have one model so talking about expected or average prediction values might seem a little strange. However, imagine you could repeat the whole model building process more than once: each time you gather new data and run a new analysis creating a new model. Due to randomness in the underlying data sets, the resulting models will have a range of predictions. Bias measures how far off in general these models' predictions are from the correct value.

## **Error due to Variance**: 
The error due to variance is taken as the variability of a model prediction for a given data point. Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model.


# **Algo** 

## **Regression**
     
### **linear regression** 

https://coursera.org/share/dc8d642a878feedbe80b1e97c9436145
![<NOM_DU_FICHIER>](../notes/img/linreg.png)


### **polynamial regression** 
Polynomial regression is not really a proper model but it's just a linear regression  with polynamial transformation on this features
https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

### **ridge regression**
La régression ridge nous permet de réduire l'amplitude des coefficients d'une régression linéaire et d'éviter le sur-apprentissage.
https://www.youtube.com/watch?v=oh4PNaT5s3c&ab_channel=Science4All
https://www.youtube.com/watch?v=Q81RR3yKn30&ab_channel=StatQuestwithJoshStarmer
https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507806-reduisez-le-nombre-de-variables-utilisees-par-votre-modele

https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization?hl=fr

![<NOM_DU_FICHIER>](../notes/img/ridge.png)

### **lasso regression** 
Il s'agit donc d'une méthode de sélection de variables et de réduction de dimension supervisée : les variables qui ne sont pas nécessaires à la prédiction de l'étiquette sont éliminées.

Attention : 
Si plusieurs variables corrélées contribuent à la prédiction de l'étiquette, le lasso va avoir tendance à choisir une seule d'entre elles (affectant un poids de 0 aux autres), plutôt que de répartir les poids équitablement comme la régression ridge. C'est ainsi qu'on arrive à avoir des modèles très parcimonieux. Cependant, laquelle de ces variables est choisie est aléatoire, et peut changer si l'on répète la procédure d'optimisation. Le lasso a donc tendance à être instable.
![<NOM_DU_FICHIER>](../notes/img/lasso.png)

https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507806-reduisez-le-nombre-de-variables-utilisees-par-votre-modele

https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization?hl=fr

### **elasticnet regression** 
L'elastic net combine les normes ℓ1 et ℓ2( ridge et lasso) pour obtenir une solution moins parcimonieuse que le lasso, mais plus stable et dans laquelle toutes les variables corrélées pertinentes pour la prédiction de l'étiquette sont sélectionnées et reçoivent un poids identique.
Le lasso peut donc être utilisé comme un algorithme de réduction de dimension supervisée.
https://openclassrooms.com/fr/courses/4444646-entrainez-un-modele-predictif-lineaire/4507806-reduisez-le-nombre-de-variables-utilisees-par-votre-modele


## **Classification** 

### **Kernel trick** 
    appliquer une transformation sur ses données pour qu'elle soit linéairement séparable.
    https://www.youtube.com/watch?v=tpNs-Hz6-LM
     Ces fonctions mathématiques permettent de séparer les données en les projetant dans un feature space (un espace vectoriel de plus grande dimension, voir figure ci-dessous)
![<NOM_DU_FICHIER>](../notes/img/kernel_trick.png)

# **SVM**

https://dataanalyticspost.com/Lexique/svm/
https://www.youtube.com/watch?v=N1vOgolbjSc&ab_channel=AliceZhao
https://www.youtube.com/watch?v=Y6RRHw9uN9o&ab_channel=AugmentedStartups

 leur principe est simple : il ont pour but de séparer les données en classes à l’aide d’une frontière aussi « simple » que possible, de telle façon que la distance entre les différents groupes de données et la frontière qui les sépare soit maximale. Cette distance est aussi appelée « marge » et les SVMs sont ainsi qualifiés de « séparateurs à vaste marge », les « vecteurs de support » étant les données les plus proches de la frontière.
![<NOM_DU_FICHIER>](../notes/img/svm.png)

En résumé : cherche à trouver la plus large marge en les vecteursde supports par l'interédiaire d'un probème d'optimisation sous contrainte (lagrange multipliers technique)

Hyper-paramêtr à retenir : 
C qui permet de décider de la pénalité des missclassifications. Un low C pourra permettre certains erreurs de classification mais une meilleur généralisation ( higher bias but lower variance?) un High C ne permettra pas d'erreur de classification (low bias, high variance)
dans le cas du kernal radial basis: alpha déterine la complexité du kernel et donc du modèle.

### **logistic Regression**

### **decision tree**

- complexification à l'infini 
- se généralise parfaitement au très grand dimension
- très rapidemment calculable
- MAIS gros risuqe d'overfitting (l'arbre peut s'agrandir jusqu'a matcher parfaitement les données)
https://www.youtube.com/watch?v=D2IazsNG9_g&list=PLtzmb84AoqRTl0m1b82gVLcGU38miqdrC&index=17&ab_channel=Science4All

### **random forest** 
ensemble d'arbre de décision (chaque arbre est différent étant données que la racine est différente)
le but étant de faire assez d'arbre pour être résistant aux fluctuations des données et limiter la variance
https://www.youtube.com/watch?v=D2IazsNG9_g&list=PLtzmb84AoqRTl0m1b82gVLcGU38miqdrC&index=17&ab_channel=Science4All


# **Scoring**

## **GLOBAL**
    https://scikit-learn.org/stable/modules/model_evaluation.html

## **Classification** :

### **Confusion matrix** 
### Accuracy 
TP + TN / (TP + TN + FP + FN)
### Precision
TP / (TP + FP)
### Recall
TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
### F1
2 * Precision * Recall / (Precision + Recall) 

https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826

### ROC and AUC (PR and ): 
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=fr#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate

https://blog.octo.com/quel-sens-metier-pour-les-metriques-de-classification/#:~:text=Precision%2DRecall%20AUC%20(PR%20AUC)&text=Plus%20une%20courbe%20a%20un,la%20performance%20globale%20du%20classifieur.

## Regression :

### **R2**
R², ou R-carré est appelé coefficient de détermination, est utilisé surtout en statistiques pour juger de la qualité d’une régression linéaire

![<NOM_DU_FICHIER>](../notes/img/r2.png)

Weakness : https://freakonometrics.hypotheses.org/75

### **RMSE**
rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predicted))


## **AIC ou BIC**