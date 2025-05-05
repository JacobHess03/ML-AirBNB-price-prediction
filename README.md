# Airbnb Price Prediction ML Pipeline

## Descrizione

Questa repository contiene uno script Python (`scripts/airbnb_price_pipeline.py`) per l'analisi esplorativa, la pulizia e la predizione dei prezzi di affitto su Airbnb. Si parte da un dataset grezzo (`data/listings.csv`), si esegue:

1. **Caricamento e pulizia dati** (rimozione di colonne non rilevanti, gestione di date e valori mancanti)
2. **Esplorazione** (statistiche descrittive, visualizzazioni con Seaborn/Matplotlib)
3. **Feature Engineering**:

   * Label Encoding di variabili categoriche (es. `neighbourhood_group`, `room_type`)
   * Creazione di variabili derivate (`price_range`, filtri IQR, ecc.)
4. **Preparazione dati per ML**:

   * Log-transform del target (`price` → `log10(price)`)
   * Split train/test (25% test)
   * Scaling selettivo con `ColumnTransformer` (solo colonne float)
5. **Training e valutazione modelli**:

   * **Modelli lineari** su dati scalati: Linear Regression, Bayesian Ridge
   * **Modelli ad albero** su dati non scalati: Decision Tree Regressor, Gradient Boosting Regressor
   * Metriche di performance: RMSE, R², MAE
   * Cross-validation (5‑fold) per Gradient Boosting
6. **Visualizzazioni finali**:

   * Matrice di correlazione post-encoding
   * Grafico Real vs Predicted per ogni modello
   * Analisi outlier (boxplot residui, rimozione IQR)

## Struttura del repository

```bash
data/                     # Dataset raw e processed
  ├── listings.csv         # Dataset originale Airbnb
scripts/                  # Script di analisi e modellazione
  └── airbnb_price_pipeline.py
notebooks/                # Analisi interattiva (EDA)
  └── eda_airbnb.ipynb
tests/                    # (opzionale) test unitari
README.md                 # Documentazione di questo progetto
requirements.txt          # Dipendenze Python
```

## Requisiti

* Python 3.8+
* pacchetti Python (vedi `requirements.txt`)

### Dipendenze principali

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
xgboost
scipy
folium (opzionale)
```

## Installazione

1. Clona il repository:

```

git clone [https://github.com/JacobHess03/ML-AirBNB-price-prediction/tree/main](https://github.com/JacobHess03/ML-AirBNB-price-prediction/tree/main)
cd airbnb-price-prediction
```

2. Crea e attiva un ambiente virtuale:
  ```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate   # Windows
  ```

3. Installa le dipendenze:

```

pip install -r requirements.txt
```


## Esecuzione dello script
```bash
python scripts/airbnb_price_pipeline.py
````

Questo comando esegue in sequenza:

* Pulizia e trasformazioni sui dati
* Analisi esplorativa e plot
* Preparazione dei dati per ML (split, scaling)
* Addestramento e valutazione di 4 modelli
* Visualizzazioni dei risultati e gestione outlier

## Personalizzazioni

* **Percorso del dataset**: Modifica `data/listings.csv` in `airbnb_price_pipeline.py` se necessario.
* **Parametri modelli**: Adatta `max_depth`, `n_estimators`, learning rate, ecc.
* **Selezione feature**: Aggiorna la lista `X_cols` per includere o escludere variabili.
* **Filtri outlier**: Cambia strategia IQR o soglie nel codice.

## Risultati attesi

* Grafici EDA (istogrammi, boxplot, scatter, mappe)
* Matrice di correlazione annotata
* Tabella di confronto RMSE/R²/MAE per ogni modello
* Plot Real vs Predicted e residui
* Report di outlier rimossi e miglioramento R²

## Contributi

Contributi, pull request e issue sono i benvenuti!
Assicurati di aprire un issue per discutere modifiche sostanziali.

## Licenza

Licenza MIT.

*Autore: Giacomo Visciotti*
