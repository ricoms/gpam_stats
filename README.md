[![DOI](https://zenodo.org/badge/91934660.svg)](https://zenodo.org/badge/latestdoi/91934660)

# gpam_stats

Implementation of statistics used for paper soon to be cited here.

This work was done in collaboration with [Lucas Chesini Okimoto](https://www.linkedin.com/in/lucas-chesini-okimoto-76646925/) and [Ana Carolina Lorena](http://lattes.cnpq.br/3451628262694747) from [ICT-UNIFESP](http://www.unifesp.br/campus/sjc/), São José dos Campos, SP - Brazil.

# Project organization 

This project is organized as below
::

    ArtificialDataset
    ├── dataFelipe1.csv
    └── dataFelipe2.csv
    logs
    └── ArtificialDataset.log
    outputs
    └── run_ArtificialDataset.csv
    scripts
    ├── __initi__.py
    ├── run_all.py
    └── stats.py
    LICENSE
    README.md
    requirements.txt
..

The main code is kept inside `scripts/stats.py` where all statistics mentioned in the article are implemented, this code is intended to be turned into a ptyhon library for a later use. The `scripts/run_all.py` is just a helper script to run all statistics on datasets kept in a specific folder, you can edit this file to test on your datasets.

The dataset used for this example is an artificial dataset available for public use if necessary. Inside folds `logs` and `outputs` are the results of running the `run_all.py` script over both datasets inside `ArtificialDataset` folder. The log will give you the time it took to calculate each estatistic for each dataset, and `outputs/` will contain a .csv file with each statistic value for each dataset.

We also provide a `requirements.txt` file for fast installation of packages required to run this project. We used Python version 3.5.2 to implement and run this code.
