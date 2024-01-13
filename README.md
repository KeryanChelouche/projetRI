# Weakly Supervised Label Smoothing
## Description

Ce projet tente de réimplémenter le papier de recherche [**Weakly Supervised Label Smoothing**](https://arxiv.org/abs/2012.08575). 

Nous tentons donc de répondre à ces deux questions: 

**RQ1** : Le lissage des étiquettes est-il un régulariseur efficace
pour les modèles neuronaux de classement de rang (L2R), et si
oui, dans quelles conditions ?

**RQ2** : Le WSLS est-il plus efficace que le LS pour
l'apprentissage des modèles neuronaux de classement de rang
(L2R) ?


## Modèle

Nous utilisons un modèle Learning to Rank basé sur BERT. 



## Negative Sampling

Le projet emploie des méthodes de Negative Sampling, y compris NS Random et NS BM25, pour sélectionner des documents non pertinents pour chaque requête.


## Techniques de Lissage des Étiquettes (LS et WSLS)

Nous explorons plusieurs techniques, notamment Cross-Entropie, T-LS (Traditional Label Smoothing), et T-WSLS (Traditional Weakly Supervised Label Smoothing).
Protocole d'Évaluation

Le protocole comprend la collecte des données, la transformation en triplets, l'apprentissage du modèle, et l'évaluation basée sur des critères spécifiques.
Résultats et Analyse

[Inclure un résumé des résultats clés et des analyses pertinentes]
Conclusion

[Conclusion sur l'efficacité du lissage des étiquettes et comparaison entre WSLS et LS]
Installation

[Instructions d'installation et de configuration]
Utilisation

[Instructions détaillées sur comment utiliser le projet]
Contribution

[Guidelines pour contribuer au projet]

N'oubliez pas d'ajouter des détails spécifiques à chaque section pour fournir des informations complètes sur votre projet.
