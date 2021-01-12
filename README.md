# [Processus Stochastiques INFOM222](https://directory.unamur.be/teaching/courses/INFOM222): Projet

Code source pour le projet à remettre le 15-01-2021.

Ce projet utilise l'interpréteur CPython 3.8.2.

## Installation

Installer les dépendances avec la commande :
```
pip install -r requirements.txt
```

## Utilisation

```
usage: main.py [-h] [--save-figures] [--w2] [seed]

Produce the different results presented in the project report.

positional arguments:
  seed            seed to initialize the random generator

optional arguments:
  -h, --help      show this help message and exit
  --save-figures  Whether to save figures or not in ./out
  --w2            Compute E[W^2] for rho = 0.5 INSTEAD of printing all the
                  graphs
```

## Note
Le rapport utilise la graine 322555291929963904604360825690372611972.

Cela a pris ~5 minutes au programme pour s'exécuter sur ma machine pour générer les graphes et ~20 pour estimer E\[W^2\] (option -w2).

## Auteur

Adrien Horgnies
