# Audit de l'Experience Replay à 20 %

Comparaison M2 / M3 / M3-R.

Une réponse contradictoire est considérée comme incorrecte par l'évaluateur V16.

## MATHS

### maths_validation_0023 — AMÉLIORATION

Question : Calcule : 17 + 33

Attendu : 50

**M2**

- Prédiction : 50
- Statut : correct
- Correct : True
- Exactitude + format : False

````text
On ajoute les nombres.
ON obtient 50.
REPONSE: 50
L'instruction a été respectée avec précision.
````

**M3**

- Prédiction : 40
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On ajoute les nombres.
On obtient 40.
REPONSE: 40
````

**M3R**

- Prédiction : 50
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On calcule de la gauche à droite.
17 + 33 = 50
REPONSE: 50
````

### maths_validation_0003 — AMÉLIORATION

Question : Calcule : (79 - 7) * 10

Attendu : 720

**M2**

- Prédiction : 100
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On soustrait 79 des 80 = 1.
On multiplie par 10 = 100.
REPONSE: 100
````

**M3**

- Prédiction : 62
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On soustrait 79 et 7.
(62) * 10 = 620
REPONSE: 62
````

**M3R**

- Prédiction : 720
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On calcule comme suit: (79 - 7) = 72 et 72 * 10 = 720.
REPONSE: 720
````

### maths_validation_0020 — RÉGRESSION

Question : Calcule : 92 - 53

Attendu : 39

**M2**

- Prédiction : 39
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait des nombres positifs, on reste avec un nombre.
REPONSE: 39
````

**M3**

- Prédiction : 39
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait des nombres.
92 - 53 = 39
REPONSE: 39
````

**M3R**

- Prédiction : 49
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule de la droite.
On ajoute 53 des 92 = 49.
REPONSE: 49
````

### maths_validation_0007 — RÉGRESSION

Question : Calcule : 44 * 17

Attendu : 748

**M2**

- Prédiction : 748
- Statut : correct
- Correct : True
- Exactitude + format : False

````text
On multiplie 44 par 17.
ON REJAUNE.
REPONSE: 748.
````

**M3**

- Prédiction : 748
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On multiplie les deux entiers.
On obtient 748.
REPONSE: 748
````

**M3R**

- Prédiction : 768
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule : 768.
REPONSE: 768
````

### maths_validation_0010 — RÉGRESSION

Question : Calcule : 160 - 149

Attendu : 11

**M2**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On soustrait des nombres, on retire l'un de l'autre. REPONSE: 11
````

**M3**

- Prédiction : 11
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait des nombres.
160 - 149 = 11
REPONSE: 11
````

**M3R**

- Prédiction : 1
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule de la droite.
On ajoute 149.
REPONSE: 1
La réponse est 1.
````

## ALGEBRE

### algebre_validation_0004 — AMÉLIORATION

Question : Résous l'équation : 8x - 18 = 166

Attendu : 23

**M2**

- Prédiction : 23
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On ajoute 18 des deux côtés.
8x = 166 + 18 = 184
On divise par 8.
x = 23
REPONSE: 23
````

**M3**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On ajoute 18 des deux côtés.
8x = 166 + 18 = 184
On divise par 8.
x = 23
REPONSE
La solution est x = 23.
L'opposition a été corrige.
````

**M3R**

- Prédiction : 23
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On ajoute 18 des deux côtés.
8x = 166 + 18 = 184
On divise par 8.
x = 23
REPONSE: 23
````

### algebre_validation_0009 — AMÉLIORATION

Question : Résous l'équation : 11x + 2 = 332

Attendu : 30

**M2**

- Prédiction : 30
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 2 des deux côtés.
11x = 330
On divise par 11.
x = 30
REPONSE: 30
````

**M3**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On soustrait 2 des deux côtés.
11x = 330
On divise par 11.
x = 30
REPONSE
La solution est x = 30.
L'opposition n'est pas nécessaire.
````

**M3R**

- Prédiction : 30
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 2 des deux côtés.
11x = 332 - 2 = 330
On divise par 11.
x = 30
REPONSE: 30
````

### algebre_validation_0014 — AMÉLIORATION

Question : Résous l'équation : 12x + 2 = 122

Attendu : 10

**M2**

- Prédiction : 10
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 2 des deux côtés.
12x = 120
On divise par 12.
x = 10
REPONSE: 10
````

**M3**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On soustrait 2 des deux côtés.
12x = 120
On divise par 12.
x = 10
REPONSE
La solution est x = 10.
L'opposition n'est pas nécessaire.
````

**M3R**

- Prédiction : 10
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 2 des deux côtés.
12x = 120
On divise par 12.
x = 10
REPONSE: 10
````

### algebre_validation_0002 — AMÉLIORATION

Question : Résous l'équation : 9x + 3 = 102

Attendu : 11

**M2**

- Prédiction : 11
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 3 des deux côtés.
9x = 102 - 3 = 99
On divise par 9.
x = 11
REPONSE: 11
````

**M3**

- Prédiction : 2
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On soustrait 3 des deux côtés.
9x = 102 - 3 = 99
On divise par 9.
REPONSE
2
````

**M3R**

- Prédiction : 11
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 3 des deux côtés.
9x = 102 - 3 = 99
On divise par 9.
x = 11
REPONSE: 11
````

### algebre_validation_0022 — AMÉLIORATION

Question : Résous l'équation : 14x + 7 = 301

Attendu : 21

**M2**

- Prédiction : 21
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 7 des deux côtés.
14x = 294
On divise par 14.
x = 21
REPONSE: 21
````

**M3**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On soustrait 7 des deux côtés.
14x = 294
On divise par 14.
x = 21
REPONSE
La solution est x = 21.
L'erreur était dans la soustraction des termes positives.
````

**M3R**

- Prédiction : 21
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On soustrait 7 des deux côtés.
14x = 301 - 7 = 294
On divise par 14.
x = 21
REPONSE: 21
````

## CODE

### code_validation_0028 — AMÉLIORATION

Question : Quel est le résultat de ce programme Python ?
print(sum([38, 73, 87]))

Attendu : 198

**M2**

- Prédiction : 200
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
200
REPONSE: 200
````

**M3**

- Prédiction : 208
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
La fonction sum() additionne les éléments de la liste.
38 + 73 + 87 = 208
REPONSE: 208
````

**M3R**

- Prédiction : 198
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
La fonction sum() additionne les éléments de la liste.
38 + 73 + 87 = 198
REPONSE: 198
````

### code_validation_0001 — RÉGRESSION

Question : Quel est le résultat de ce programme Python ?
print(len([36, 80, 88, 79, 16, 17, 13, 51, 48, 44]))

Attendu : 10

**M2**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
Le programme calcule la longueur des éléments inclus dans la liste [36, 80, 88, 79, 16, 17, 13, 51, 48, 44]. Il y a 11 éléments, donc le résultat est 11. REPONSE: 11
````

**M3**

- Prédiction : 10
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
La fonction len() retourne le nombre d'éléments.
La liste contient 10 éléments.
REPONSE: 10
````

**M3R**

- Prédiction : 11
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
La fonction len() retourne le nombre d'éléments.
La liste contient 11 éléments.
REPONSE: 11
````

### code_validation_0006 — RÉGRESSION

Question : Quel est le résultat de ce programme Python ?
print(7**2 + 15)

Attendu : 64

**M2**

- Prédiction : 29
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On multiplie 7 par 2 et on ajoute 15.
print(7*2 + 15) = print(14 + 15) = print(29)
REPONSE: 29
````

**M3**

- Prédiction : 64
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On calcule first.
7**2 = 49
49 + 15 = 64
REPONSE: 64
````

**M3R**

- Prédiction : 97
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule la puissance et la somme.
Resultat = 97
REPONSE: 97
````

### code_validation_0014 — RÉGRESSION

Question : Quel est le résultat de ce programme Python ?
print(6**4 + 7)

Attendu : 1303

**M2**

- Prédiction : 35
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On multiplie 6 par 4 et on ajoute 7.
print(6*4 + 7) = 28 + 7 = 35
REPONSE: 35
````

**M3**

- Prédiction : 1303
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On calcule d'abord la puissance.
6**4 = 1296
On ajoute 7.
1296 + 7 = 1303
REPONSE: 1303
````

**M3R**

- Prédiction : 10007
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule la puissance et la somme.
Resultat = 10000 + 7 = 10007
REPONSE: 10007
````

### code_validation_0003 — RÉGRESSION

Question : Quel est le résultat de ce programme Python ?
print(7**4 + 14)

Attendu : 2415

**M2**

- Prédiction : 
- Statut : missing
- Correct : False
- Exactitude + format : False

````text
On multiplie 7 par 4 et on ajoute 14.
ONREPSONSE: 56
````

**M3**

- Prédiction : 2415
- Statut : correct
- Correct : True
- Exactitude + format : True

````text
On calcule d'abord la puissance.
7**4 = 2401
On ajoute 14.
2401 + 14 = 2415
REPONSE: 2415
````

**M3R**

- Prédiction : 205
- Statut : incorrect
- Correct : False
- Exactitude + format : False

````text
On calcule la puissance et la somme.
La réponse est 205.
REPONSE: 205
````
