# DÔLEŽITÉ!
Na spustenie projektu musí byť nainštalovaný conda
https://anaconda.org/anaconda/conda

Taktiež je potrebné mať nainštalovaný CUDA Toolkit (v projekte bola použitá verzia 12.6)
https://developer.nvidia.com/cuda-toolkit


# INŠTALÁCIA PROSTREDIA:
Pre PyTorch 2.7 a CUDA 12.6:
V termináli v priečinku s projektom zadaj:
> conda env create -f environment.yml

> conda activate IGEV_cupy

Následne nainštaluj PyTorch podľa tvojej verzie CUDA Toolkitu. Príkaz pre konkrétnu verziu nájdeš na:  
https://pytorch.org/

Príklad pre CUDA 12.6:
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Musíš tiež nainštalovať presne túto verziu knižnice `timm`, aby správne fungovala neurónová sieť IGEV++:
> pip install timm==0.5.4


# SPUSTENIE PROJEKTU:
Spusti v priečinku s projektom:
> python main.py --left dataset/1/2/left.png --right dataset/1/2/right.png --resize 0.333 --output output/middle.png --model RAFT

# DOSTUPNÉ PARAMETRE:
| Parameter   | Popis                                                             |
| ----------- | ----------------------------------------------------------------- |
| `--left`    | Cesta k ľavému alebo spodnému obrázku *(povinné)*                 |
| `--right`   | Cesta k pravému alebo hornému obrázku *(povinné)*                 |
| `--resize`  | Resize faktor (napr. `0.333`) *(predvolené: 1)*                   |
| `--model`   | Model na výpočet disparity *(predvolené: IGEV)* - `IGEV` / `RAFT` |
| `--topdown` | Pridať, ak ide o horný a dolný obrázok                            |
| `--output`  | Výstupná cesta a názov výsledného obrázku     


# PRÍKLAD PRE 4 KAMERY:
> python main.py --left dataset/dataset/BL.jpeg --right dataset/dataset/BR.jpeg --resize 0.333 --output output/B5.png

> python main.py --left dataset/dataset/TL.jpeg --right dataset/dataset/TR.jpeg --resize 0.333 --output output/T5.png

> python main.py --left output/B5.png --right output/T5.png --resize 1 --output output/center.png --topdown

# PRÍKLAD PRE HORNÝ A DOLNÝ OBRAZOK:
> python main.py --left dataset/dataset/BL.jpeg --right dataset/dataset/TL.jpeg --resize 0.333 --output output/L5.png --topdown

> python main.py --left dataset/dataset/BR.jpeg --right dataset/dataset/TR.jpeg --resize 0.333 --output output/T5.png --topdown

# PRÍKLAD PRE POUŽITIE INÉHO MODELU:
> python main.py --left dataset/1/2/left.png --right dataset/1/2/right.png --resize 0.333 --output output/middle.png --model RAFT