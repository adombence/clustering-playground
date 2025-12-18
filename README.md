## Clustering Playground

Kísérleti játszótér klaszterező algoritmusokhoz. A projekt célja, hogy különböző elterjedt (és egy fuzzy) klaszterező módszereket egységes környezetben próbáljunk ki, mérjük és összehasonlítsuk őket szintetikus adatokon, főként a make_moons példán, skálázás után.

### 📦 Főbb elemek

- **Algoritmusok**: K-Means, K-Medoids, Agglomeratív (Ward/Complete/Average), DBSCAN, Fuzzy C-Means (FCM)
- **Metrikák**: Silhouette, Davies–Bouldin, Calinski–Harabasz + futási idő
- **Jegyzetfüzetek**: Lépésről-lépésre bemutatás interaktív környezetben
- **Rácskeresés**: Egyszerű hiperparaméter-keresés és összegző táblák CSV-ben


## Követelmények

- Python 3.10+ ajánlott
- Függőségek a `requirements.txt` alapján:
	- numpy, scipy, pandas, matplotlib, seaborn
	- scikit-learn, scikit-learn-extra
	- fuzzy-c-means
	- notebook (Jupyter)


## Telepítés

Ajánlott egy külön virtuális környezet használata.

```zsh
# Klónozás után lépj be a mappába
cd clustering-playground

# (Opcionális) virtuális környezet
python3 -m venv .venv
source .venv/bin/activate

# Függőségek telepítése
pip install -r requirements.txt
```


## Mappa-struktúra

- `notebooks/` – lépésenkénti bemutatók és összehasonlítás
	- `01_datasets_and_scaling.ipynb` – szintetikus adatok, skálázás
	- `02_kmeans_kmedoids.ipynb` – K-Means és K-Medoids
	- `03_agglomerative.ipynb` – Agglomeratív klaszterezés (Ward/Complete/Average)
	- `04_dbscan.ipynb` – DBSCAN
	- `05_fuzzy_cmeans.ipynb` – Fuzzy C-Means (FCM)
	- `06_compare.ipynb` – összegző/összehasonlító áttekintés
- `src/` – futtatható kód és segédfüggvények
	- `algorithms.py` – algoritmusok egységes futtatása
	- `evaluation.py` – belső (labelt nem igénylő) metrikák
	- `run_experiments.py` – rácskeresés és eredmények mentése
- `results/tables/` – CSV eredménytáblák (összefoglalók, legjobbak, stb.)


## Jegyzetfüzetek futtatása

Megnyithatod a jegyzetfüzeteket VS Code-ból vagy Jupyterből.

```zsh
# Jupyter indítása (ha nem VS Code-ot használsz)
jupyter notebook
```

Ezután lépj a `notebooks/` mappába, és futtasd a cellákat sorban. A notebookok ugyanazokat az algoritmusokat és skálázási lépéseket használják, mint a szkript.


## Kísérletek futtatása parancssorból

A teljes rácskeresést a `src/run_experiments.py` végzi. Ez a szkript:
- 800 mintás `make_moons` adatot generál, majd `StandardScaler`-rel skáláz
- több algoritmust és paraméterrácsot próbál végig
- metrikákat számol: Silhouette (`sil`), Davies–Bouldin (`db`), Calinski–Harabasz (`ch`), valamint időt (`time`)
- mindent CSV-be ment a `results/tables/summary.csv` fájlba

Futtatás a projekt gyökeréből:

```zsh
# A szkript közvetlen futtatása (ajánlott)
python src/run_experiments.py

# Alternatíva: a src mappából
cd src
python run_experiments.py
```

Megjegyzés: a `python -m src.run_experiments` hívás a jelenlegi importok miatt nem ajánlott.


## Algoritmusok és paraméterrács

A `src/run_experiments.py` a következő rácsot vizsgálja (random_state=42):

- K-Means: `k` ∈ {2..7}
- K-Medoids: `k` ∈ {2..7}
- Agglomeratív: `k` ∈ {2..7}, `link` ∈ {"ward", "complete", "average"}, `metric` = "euclidean"
- DBSCAN: `eps` ∈ {0.1, 0.3, 0.5, 0.7, 1.0}, `min_samples` ∈ {3, 5}
- Fuzzy C-Means (FCM): `k` ∈ {2..7}, `m` = 2.0

Az FCM esetén a címkék a tagsági mátrix (`model.u`) argmax-ából keletkeznek.


## Metrikák

- Silhouette (`sil`): magasabb jobb ([-1, 1])
- Davies–Bouldin (`db`): alacsonyabb jobb (≥ 0)
- Calinski–Harabasz (`ch`): magasabb jobb (≥ 0)
- Futási idő (`time`): másodperc

Szélsőséges esetekben (pl. minden pont zaj DBSCAN-nél) NaN értékek kerülhetnek a metrikákba.


## Eredmények

- Összegző tábla: `results/tables/summary.csv`
- Egyéb táblák (jegyzetfüzetekből):
	- `results/tables/kmeans_kmedoids.csv`
	- `results/tables/agglomerative.csv`
	- `results/tables/dbscan.csv`
	- `results/tables/compare_best.csv`

Az összefoglaló futás végén a konzolra egy gyors átlagolt rangsor is kikerül (silhouette szerinti rendezéssel).

### 📊 Példa eredmények

A `06_compare.ipynb` notebook összehasonlítja az összes algoritmus legjobb paraméterbeállítását:

| Algoritmus | Silhouette ↑ | Davies-Bouldin ↓ | Futási idő (s) |
|------------|--------------|------------------|----------------|
| K-Means    | ~0.55        | ~0.65            | ~0.02          |
| DBSCAN     | ~0.50        | ~0.75            | ~0.01          |
| Agglo-Ward | ~0.54        | ~0.68            | ~0.03          |

**Megjegyzés**: Az eredmények a `make_moons` szintetikus adatokon alapulnak, 800 mintával.


## Reprodukálhatóság

- A véletlenszám-generátor magja (`random_state`) 42 minden érintett komponensnél
- Az adatok skálázása `StandardScaler`-rel történik


## Hasznos tippek


### ⚠️ Ismert problémák (Known Issues)

- **DBSCAN zaj problémák**: Bizonyos `eps` értékeknél az összes pont zaj lehet (minden címke -1), ami NaN metrikákat eredményez. 
	- **Megoldás**: Használd a k-distance plot-ot (`04_dbscan.ipynb`) az optimális `eps` kiválasztásához.
  
- **FCM konvergencia**: Nagy `m` értékek (pl. m > 3) lassú konvergenciát okozhatnak.
	- **Megoldás**: Maradj az `m=2.0` körüli értékeknél.

- **Notebook kimenetek**: A notebookok base64-kódolt képeket tartalmazhatnak, ami nagy fájlméretet okoz.
	- **Megoldás**: Használd az `nbstripout` eszközt commit előtt (telepítés: `pip install nbstripout && nbstripout --install`).


## Licenc

Ez a projekt MIT licenc alatt áll. Lásd a `LICENSE` fájlt a részletekért.

