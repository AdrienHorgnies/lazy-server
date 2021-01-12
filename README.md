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
Le rapport utilise la graine 63605269805769599628681422967949126226.

Cela a pris X minutes au programme à s'exécuter sur ma machine.

## Auteur

Adrien Horgnies
