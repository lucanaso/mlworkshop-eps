# Workshop introduttivo sul Machine Learning, in Python
###### *di Luca Naso, organizzato da EPS Young Minds Catania*

Questo documento descrive schematicamente il contenuto del workshop, tenutosi in 2 giornate l'8 ed il 15 maggio 2021.
- Il **codice** relativo al workshop è presente in questo stesso [repository](https://github.com/lucanaso/mlworkshop-eps/tree/master/code): github.com/lucanaso/mlworkshop-eps/tree/master/code
- Le **slides** sono invece disponibili su slideshare a questo [link](https://www.slideshare.net/LucaNaso/workshop-introduttivo-al-machine-learning-in-python): www.slideshare.net/LucaNaso/workshop-introduttivo-al-machine-learning-in-python.

## Agenda
Il workshop alterna parti di lezioni teoriche con slide, a parti di pratica con codice scritto dal vivo, secondo il seguente programma:

1. [Introduzione al ML](#1-introduzione-al-ml-slides) (slides)
2. [Creazione del Dataset in Pytho](#2-dataset-creation-in-python-codice) (codice)
3. [Problemi di Regressione](#3-problemi-di-regressione) (slides e codice)
4. [Valutazione dei Modelli](4-valutazione-dei-modelli) (slides e codice)
5. [Problemi di Classificazione](5-problemi-di-classificazione) (codice)
6. [Problemi di Clustering](6-problemi-ci-clustering) (codice)

## 1. Introduzione al ML (slides)
1. Definizione
2. Esempi
3. Tassonomia
   
    3.1 per tipo di apprendimento
   
    3.2 per tipo di output

4. Modelli di Machine Learning

Vedi la presentazione (dall'inizio fino alla slide 47).

## 2. Dataset creation in Python (codice)

1. Creazione di un dataset di N osservazioni, con 1 feature ed il relativo target.
2. I valori della feature sono creati a caso usando una distribuzione omogenea.
3. I valori del target sono costruiti partendo da una relazione lineare a cui si aggiunge un livello di variabilità usando una distribuzione gaussiana standard.
4. I dati sono poi riportati in un grafico scatterplot (X, y).

Si usano:
- *numpy.random* per la generazione del dataset 
- *matplotlib.pyplot* per il grafico
  
Vedi il codice al file [live_coding_2021_05_08.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/live_coding_2021_05_08.py)

## 3. Problemi di Regressione
### 3.1 Simple Linear Regression - SLR
Per le slide:
1. Quando si usa la SRL
2. Interpretazione grafica della SRL
3. Residuo, RSS e metodo dei minimi quadrati

Vedi presentazione (slide 50 -> 62)

Per il codice:
1. Creazione dell'oggetto *LinearRegression* (classe di *sklearn.linear_model*)
2. Uso del metodo fit
3. Uso del metodo predict
4. Confronto dei coefficenti del modello con quelli della sorgente dei dati

Vedi codice al file [live_coding_2021_05_15.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/live_coding_2021_05_15.py#L30-L60) (al 15/05/2021, 2o commit, vedi le righe 30-60).


### 3.2 Multiple Linear Regression - MLR
Seguiamo lo stesso percorso della SLR.

Vedi presentazione (slide 63 -> 69).

Vedi codice al file [live_coding_2021_05_15.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/live_coding_2021_05_15.py#L62-L133) (al 15/05/2021, 2o commit, vedi le righe 62-133).

## 4. Valutazione dei modelli
Per le slide:
1. Definizione del problema
2. Validation set
3. LOOCV
4. K-fold CV

Vedi presentazione (slide 63 -> 69)

Per il codice:
1. Usiamo *cross_validate* da *sklearn.model_selection*
2. Impostiamo 5 fold (K = 5)
3. e scegliamo come metrica l'MSE (score='neg_mean_squared_error')


Vedi codice al file [live_coding_2021_05_15.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/live_coding_2021_05_15.py#L134-L143) (al 15/05/2021, 2o commit, vedi le righe 134-143).

Vedi anche il file [full_code.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/full_code.py#L289-L411) (al 15/05/2021, 2o commit, vedi le righe 289-411).

## 5. Problemi di Classificazione
Non trattato durante il workshop.

Le slide sono comunque presenti nella presentazione (83-84). 

Il codice è inserito nel file [full_code.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/full_code.py#L413-L451) (al 15/05/2021, 2o commit, vedi le righe 413-451).

## 6. Problemi di Clustering
Non trattato durante il workshop.

Le slide sono comunque presenti nella presentazione (85-86).

Il codice è inserito nel file [full_code.py](https://github.com/lucanaso/mlworkshop-eps/blob/master/code/full_code.py#L453-L532) (al 15/05/2021, 2o commit, vedi le righe 453-532).


