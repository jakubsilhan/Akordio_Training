# Akordio Train
Tento repozitář obsahuje kód pro trénování modelů pro rozpoznávání akordů a jejich vyhodnocení.

## Zprovoznění projektu
Následují kroky pro zprovoznění projektu.

  1) Neprve je nutné naklonovat repozitář i se submodulem.
      - `git clone git@github.com:jakubsilhan/Akordio_Training.git --recurse-submodules`
  2) Instalace balíčků je možná přes jejich výpis v requirements.txt.
      -  `pip install -r requirements.txt`

### Požadavky
Zde jsou sepsány různé softwarové požadavky, které je nutné manuálně doinstalovat.

#### Pyrubberband
Předzpracování dat využíva program Rubber Band k provádění datových augmentací přes změny výšek tónů. K použití v Pythonu je využit wrapper pyrubberband, ale instalaci základního programu je nutné provést manuálně. Tato sekce obsahuje instrukce k provedení této instalace na Windows zařízení.

1. Stáhnout spustitelný soubor v komprimovaném formátu ze stránky https://breakfastquay.com/rubberband/
2. Extrahovat soubory do C:\Program Files\RubberBand
3. Přidat tento adresář do systémové proměnné PATH v  __Control Panel → System → Advanced system settings → Environment Variables__
4. Vyzkoušet instalaci příkazem `rubberband --help`
5. Nainstalovat Python balíček `pip install pyrubberband`

#### Pytorch
Použití jednoduchého příkazu `pip install torch` provede instalaci pouze CPU verze knihovny PyTorch. Tato instalace je defaultně instalována i přes requirements.txt, jelikož je univerzálně použitelná. Pro použití grafické karty je však nutné provést specifickou instalaci, kterou lze nakonfigurovat na stránce https://pytorch.org/get-started/locally/.

### Data
Data používaná ke trénování je nutné připravit do následující struktury souborů:
```
dataset/
|-- Chords/
|   |-- album01/
|   |   |-- song01.lab
|   |   `-- song02.lab
|   `-- album02/
|       `-- ...
|-- Audio/
|   |-- album01/
|   |   |-- song01.mp3
|   |   `-- song02.mp3
|   `-- album02/
|       `-- ...
```
Anotace musí být zapsány ve formátu `.lab`, který je zapsán následující formou:
```
0.364558	2.531134	N
2.531134	4.088050	G:maj
4.088050	5.661429	G:min
5.661429	7.254762	G:maj
7.254762	8.785193	G:min
```

Samotnou cestu k datům lze nastavit v konfiguraci trénování, která je popsána dále v sekci příkladů.

## Postupy práce
Tato kapitola obsahuje popis postupů práce s projektem. Je zde shrnuto předzpracování dat a trénování, testování a vyhodnocení modelů.  
Spouštění těchto skriptů je prováděno příkazem `python -m Scripts.<skript> -<možnosti>`.
> [!TIP]
> Informace o jednotlivých způsobech spuštění lze zobrazit možností `-h`

### Příprava dat
Předzpracování dat je spouštěno přes skript `preprocess` a využívá datovou část hlavního konfiguračního souboru.
```
python -m Scripts.preprocess
```
> [!Warning]
> Tento skript nemá žádné další nastavení

### Trénování
Trénování je spouštěno skriptem `train`, který využivá zbytek hlavního konfiguračního souboru a jednotlivé možnosti slouží k výběru způsobu trénování.
```
python -m Scripts.train
```

```
ModelTrainer [-h] [-c] [-f] [-m] [-e EPOCHS]

Program for training chord recognition models

options:
  -h, --help                    show this help message and exit
  -c, --crf                     Whether to train the CRF part of the model
  -f, --final                   Whether to train the final model
  -m, --multitask               Whether to use multitask training
  -e EPOCHS, --epochs EPOCHS    Number of epochs for the final run
```

>[!Warning]
> Všechny varianty skriptu jsou zcela závislé na hlavním konfiguračním souboru. To znamená, že trénování CRF modelu nad již vytrénovaným jiným modelem vyžaduje zachování zcela identického konfiguračního souboru. CRF model je poté uložen do stejného adresáře, akorát s jiným prefixem. 

### Testování
Testování je spouštěno skriptem `test`, který testuje model vybraný přes parametry `-m` a `-f`. Testování využívá konfiguračního souboru uloženého u natrénovaného modelu.
```
python -m Scripts.test
```

```
ModelTester [-h] [-c] [-t] -m MODEL -f FOLD

Program for testing chord recognition models

options:
  -h, --help                show this help message and exit
  -c, --crf                 Whether to test the model with its CRF layer
  -t, --test                Run final evaluation on test set
  -m MODEL, --model MODEL   Name of the model to test
  -f FOLD, --fold FOLD      Number of the validation fold
```

### Agregace výsledků
Pro agregaci výsledků daného modelu byl zprovozněn dodatečný skript `aggregate`, který vezme výsledky všech foldů z křížové validace a shrne je v jednom souboru. 
Skriptu stačí zadat název modelu a jaké výsledky dohledávat (základní/s CRF) a výsledky budou uloženy do adresáře modelu v souboru `(crf_)aggregated_data.json`.
```
python -m Scripts.aggregate
```

```
Aggrregator [-h] -m MODEL [-c]

Program for aggregating testing results

options:
  -h, --help                show this help message and exit
  -m MODEL, --model MODEL   Name of the model to aggregate
  -c, --crf                 Whether to check the crf data
```

## Obsah
Celý projekt se skládá z následujících adresářů:
- __Config_Templates:__ Vzory konfiguračních souborů pro trénování modelů
- __Evaluation:__  Skripty pro tvorbu grafů a tabulek porovnání modelů
- __Neural_Nets:__ Implementace používaných modelů v knihovně PyTorch
- __Scripts:__ Vstupní body pro práci s projektem
- __Services:__ Adresář service tříd, který obsahuje pouze třídu pro načítání předzpracovaných datových sad
- __Testers:__ Různorodé třídy pro testování natrénovaných modelů
- __Trainers:__ Různorodé třídy pro trénování modelů
- __Utils:__ Dodatkové užitečné metody

## Modely
Sekce shrnující používané modely.

### Logistická regrese
Logistická regrese zprozovněná přes knihovnu: [Sklearn](https://scikit-learn.org/stable/).

### Konvoluční síť (CNN)
Čistě konvoluční síť založená na článku: [A Fully Convolutional Deep Auditory Model for Musical Chord Recognition](https://doi.org/10.1109/MLSP.2016.7738895).


### Konvoluční Rekurentní Síť (CRNN)
Konvoluční rekurentní síť založená na článku: [Structured Training for Large-Vocabulary Chord Recognition](https://brianmcfee.net/papers/ismir2017_chord.pdf).  

>[!Note]
> Model zde existuje ve dvou variantách CR1 a CR2 s rozdílným dekodérem.

### Obousměrný transformer pro rozpoznávání akordů (BTC)
Transformerový model založený a implementovaný autory článku: [A Bi-Directional Transformer for Musical Chord Recognition](https://github.com/jayg996/BTC-ISMIR19)-

## Příklady
Tato kapitola obsahuje příklady různých souborů.

### Konfigurace
Takto vypadá konfigurační soubor, který je používaný pro předzpracování dat a trénování modelů. Jeho kopie je dodána do adresáře předzpracovaných dat i natrénovaného modelu pro účely dalšího použití.
```
base:
  random_seed: 42
data:
  dataset_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen, RobbieWilliams]
  preprocessed_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Data/BaseDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    test_split: 0.2
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6) or (216/36=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (frame_length scales with frequency -> lower f means larger window)
    fragment_size: 108 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs
    fragment_hop: 0.5
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 5 # highest pitch shift
    sampling_rate: 22050 # sampling rate
train: # used for both training a evaluating
  data_source: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Data/BaseDataset"
  model_name: "cr2_144_majmin_multi_final"
  val_fold: -1 # which fold to use as validation set
  model_type: "CR2" # CR1/CR2/BTC/CNN/LOG...
  model_complexity: "majmin" # which type of chords to train (majmin/majmin7/complex)
  checkpoint_interval: 10 # after how many epochs to checkpoint
  model:
    batch_size: 128
    input: 144 # feature size 144/216
    output: 25 # output size (number of chords 25/61/170)
    hidden: [256] # hidden size (depends on a model) 128/256
    dropout: [0.5] # dropout (depends on a model)
    layers: 1 # number of layers
    bidirectional: True
    padding_index: -1
    epoch_count: 100
    loss_patience: 10
    learning_rate: 1e-4
    loss_delta: 1e-3
    weight_decay: 1e-5
```

### Datové Sady
Zde jsou dva příklady konfigurací předzpracování dat. Jedná se o část dříve zmíněného konfiguračního souboru.
#### CQT Dataset
Tato datová sada využivá jako příznak CQT spektrum.
```
data:
  dataset_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen, RobbieWilliams]
  preprocessed_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Data/BaseDataset"
  preprocess:
    pcp:
      enabled: False # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    test_split: 0.2
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6) or (216/36=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (frame_length scales with frequency -> lower f means larger window)
    fragment_size: 108 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs ((1024/22050)*216=10.03)
    fragment_hop: 0.5
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 5 # highest pitch shift
    sampling_rate: 22050 # sampling rate
```

#### PCP Dataset
Tato datová sada využivá jako příznak PCP vektor.
```
data:
  dataset_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Datasets"
  datasets: [Beatles, Uspop2000, CaroleKing, Queen, RobbieWilliams]
  preprocessed_dir: "D:/Development/Source/University/Diploma/Akordio/Akordio-Training/Data/PCPDataset"
  preprocess:
    pcp:
      enabled: True # whether to use pcp or log CQT
      bins: 12 # frequency resolution of CQT (bin for each semitone)
      octaves: 6 # number of octaves (separately specified for chroma)
    test_split: 0.2
    num_splits: 5 # number of k-fold folds
    bins_per_octave: 24 # frequency resolution of CQT
    cqt_bins: 144 # number of total bins that specifies number of octaves (144/24=6) or (216/36=6)
    hop_length: 2048 # number of samples between consecutive analysis frames (frame_length scales with frequency -> lower f means larger window)
    fragment_size: 108 # number of frames per fragment ((2048/22050)*108=10.03s) - 0 for full songs ((1024/22050)*216=10.03)
    fragment_hop: 0.5
    pitch_shift_start: -6 # lowest pitch shift
    pitch_shift_end: 5 # highest pitch shift
    sampling_rate: 22050 # sampling rate
```
