# -*- coding: utf-8 -*-
"""
# **Utilisation de l'algorithme A PRIORI/ASSOCIATION RULES:**
"""

import pandas as pd
import pymongo as pymongo

df = pd.read_csv(r'./DATAQUIZ.csv')
Central = pymongo.MongoClient("mongodb+srv://m001-student:testtest@testcluster.yhrmt.mongodb.net/myDatabase?retryWrites=true&w=majority")
Local = pymongo.MongoClient("mongodb+srv://fares:test@cluster0.csscd.gcp.mongodb.net/LocalDB?retryWrites=true&w=majority")
dbCentral = Central.myDatabase
dbLocal = Local.LocalDB
a = list(dbCentral.Client.find({}, {'_id.ObjectId': 1}))
ids = []
for id in a:
    ids.append(str(id)[18:42])
ids.sort()
categories = ['View', 'Events', 'Decoration', 'Music', 'Flower', 'Meal', 'Desert', 'Drinks', 'Sport', 'Fruits']
keys = ['user_id',
        'vue_sur_mer', 'vue_sur_piscine', 'vue_sur_montagne', 'vue_sur_ville',
        'évènements_ambiance', 'nature', 'piscine', 'plage',
        'décoration_vintage', 'décoration_tropical', 'décoration_confort', 'décoration_éclatante',
        'musique_classique', 'music_pop', 'music_rock', 'music_jazz_blues',
        'fleur_camomille', 'fleur_tulipe', 'fleur_Clématite', 'Rose_branchue',
        'végétarien', 'poulet', 'viande', 'poisson',
        'tarte', 'glace', 'fruits_frais', 'chocolat',
        'vin', 'mokhito_jus', 'café', 'thé',
        'golf', 'tennis', 'yoga', 'hand',
        'Orange', 'Kiwi', 'Banane', 'Pomme']
clients = pd.DataFrame(columns=keys)

b = list(dbCentral.Responses.find({}, {'_id': 0, 'answers': 1, 'clientId': 1}).sort("clientId"))

for i in range(len(ids)):
    prefs = [ids[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]
    b1 = list(filter(lambda pref: pref["clientId"] == ids[i], b))
    j = 1
    for el in b1[:10]:
        for k in range(4):
            if el["answers"][k] == 1:
                prefs[j] = 1
            j = j + 1
    clients.loc[i] = prefs
basefinal = pd.concat([df, clients], ignore_index=True)
print(clients)

print(basefinal.columns)

test = basefinal.iloc[:, 1:42]

"""**Support:** c'est la popularité par défaut d'un article. En termes mathématiques, le support de l'élément A n'est rien d'autre que le rapport des transactions impliquant A au nombre total de transactions.
Support (plage) = (transactions impliquant de plage) / (transaction totale)
**Confidence**: probabilité que le client qui a aimé à la fois A et B.Il divise le nombre de transactions impliquant à la fois A et B par le nombre de transactions impliquant B.
Confidence (A => B) = (Transactions impliquant à la fois A et B) / (Transactions impliquant uniquement A)
**Lift**: Augmentation de la vente de A lorsque vous vendez B.
Lift (A => B) = Confidance (A, B) / Support (B)
Ainsi, la probabilité qu’un client achète à la fois A et B ensemble est multipliée par la «valeur d’amélioration» par rapport à la probabilité d’acheter seul.
Lift (A => B) = 1 signifie qu'il n'y a pas de corrélation dans l'ensemble d'éléments.
Lift (A => B)> 1 signifie qu'il existe une corrélation positive au sein de l'ensemble d'articles, c'est-à-dire que les produits de l'ensemble d'articles, A et B, sont plus susceptibles d'être achetés ensemble.
**Les algorithmes basés sur des règles d'association sont considérés comme une approche en deux étapes:**
**Génération d'ensembles d'articles fréquents:** recherchez tous les ensembles d'articles fréquents avec prise en charge> = nombre min_support prédéterminé
**Génération de règles**: liste toutes les règles d'association des ensembles d'éléments fréquents. Calculez le support et la confiance pour toutes les règles. Élaguez les règles qui échouent aux seuils min_support et min_confidence.
"""

# importation de la fonction apriori
from mlxtend.frequent_patterns import apriori

# itemsets frequents
freq_itemsets = apriori(test, min_support=0.025, max_len=15, use_colnames=True)
# support:indicateur de fiabilité de la règle


# nombre d'itemsets
print(freq_itemsets.shape)


# fonction de test d'inclusion
def is_inclus(x, items):
    return items.issubset(x)


# recherche des index des itemsets correspondant à une condition
import numpy

id = numpy.where(freq_itemsets.itemsets.apply(is_inclus, items={'nature'}))
print(id)

# affichage des itemsets corresp.
print(freq_itemsets.loc[id])

# itemsets contenant vue sur piscine - passer par les méthodes natives de Series
print(freq_itemsets[freq_itemsets['itemsets'].ge({'vue_sur_piscine'})])

# itemsets contenant vue_sur_piscine et tennis
print(freq_itemsets[freq_itemsets['itemsets'].ge({'tarte', 'tennis'})])

# fonction de calcul des règles
from mlxtend.frequent_patterns import association_rules

# génération des règles à partir des itemsets fréquents
regles = association_rules(freq_itemsets, metric="confidence", min_threshold=0)

print(type(regles))

# dimension
print(regles.shape)

# liste des colonnes
print(regles.columns)

# 7 "premières" règles
print(regles.iloc[:7, :])

# règles en restreignant l'affichage à qqs colonnes
myRegles = regles.loc[:, ['antecedents', 'consequents', 'lift']]
print(myRegles.shape)
# pour afficher toutes les colonnes
import pandas as pd

pd.set_option('display.max_columns', 5)
pd.set_option('precision', 3)
# affichage des 5 premières règles
print(myRegles[:5])

# trier les règles dans l'ordre du lift décroissants - 10 meilleurs règles
print(myRegles.sort_values(by='lift', ascending=False)[:10])

# filtrer les règles menant au conséquent {décoration_éclatante}
print(myRegles[myRegles['consequents'].eq({'décoration_éclatante'})])

print(myRegles['antecedents'])

from numpy import random
from bson.objectid import ObjectId
from bson.dbref import DBRef

for ind in clients.index:
    l = []
    cats = []
    for i in range(1, 40):
        if clients.loc[ind][i] == 1:
            l.append(clients.columns[i])
            index = keys.index(clients.columns[i]) - 1
            cats.append(categories[int(index / 4)])
    for i in range(len(l)):
        pref = dbCentral.Preference.find_one({"clientId": clients.loc[ind][0], "key": cats[i]})
        if pref:
            if l[i] not in list(pref["values"]):
                dbCentral.Preference.update_one(
                    {"clientId": clients.loc[ind][0], "key": cats[i]},
                    {"$push": {"values": l[i]}}
                )
        else:
            pref_id = dbCentral.Preference.insert({"key": cats[i], "clientId": clients.loc[ind][0], "values": [l[i]]})
            ref = DBRef(collection='Preference', id=pref_id)
            dbCentral.Client.update_one(
                {'_id': ObjectId(clients.loc[ind][0])},
                {'$push': {'essentials': ref}}
            )
for ind in clients.index:
    l = []
    k = 0
    for i in range(1, 40):
        if clients.loc[ind][i] == 1:
            l.append(clients.columns[i])
    if len(list(dbCentral.Preference.find({"clientId": clients.loc[ind][0]}))) > 3:
        k = 3
    for el in dbCentral.Preference.find({"clientId": clients.loc[ind][0]})[k:]:
        for val in el['values']:
            l.append(val)
    d = random.randint(len(l))
    t = {l[d]}
    if len(t) != 0:
        pr = {list(
            myRegles[myRegles['antecedents'].eq(t)].sort_values(by='lift', ascending=False)['consequents'].iloc[0])[0]}
        print(pr)
        if len(pr) != 0:
            index = keys.index(list(pr)[0]) - 1
            cat = categories[int(index / 4)]
            print(cat)
            pref = dbCentral.Preference.find_one({"clientId": clients.loc[ind][0], "key": cat})
            if pref:
                if list(pr)[0] not in list(pref["values"]):
                    dbCentral.Preference.update_one(
                        {"clientId": clients.loc[ind][0], "key": cat},
                        {"$push": {"values": list(pr)[0]}}
                    )
            else:
                pref_id = dbCentral.Preference.insert({"key": cat, "clientId": clients.loc[ind][0], "values": list(pr)})
                ref = DBRef(collection='Preference', id=pref_id)
                dbCentral.Client.update_one(
                    {'_id': ObjectId(clients.loc[ind][0])},
                    {'$push': {'essentials': ref}}
                )