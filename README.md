# bored

An attempt to use [Born's Rule](https://bornrule.eguidotti.com) for Entity Disambiguation.

## Running the code

The code uses `python 3.12.4`. To run it, do the following:

1. (Optional) Set up a virtual environment.
2. Install all the requirements (i.e, `pip install -r requirements.txt`).
3. Run the `construct_wiki_data.py` script.
4. Run the jupyter notebook that most interests you!

## Structure

### Directories

- `data/` is used to store all the data for the project;
- `models/` contains definitions of the ML models used in the notebooks;
- `runs/` contains saved models, vectorizers and encoders obtained while training;
- `utils/` contains various utility functions;
- `wiki_tools/` contains code used to interface with Wikipedia in order to download the various text snippets.

### Notebooks

- `born_ensembles.ipynb` is a notebook I used to experiment with the possibility of constructing ensembles of Born Classifiers;
- `born_layers.ipynb` contains experiments with Born Layers, in an attempt to construct a Born MLPA;
- `born_on_conll.ipynb` is a notebook that mostly served as a quick test to familiarise myself with the various pieces in play by using the Born Classifier for NER classification on the CoNLL dataset;
- `entity_disambiguation.ipynb` is the "main" notebook, i.e., the one which contains the complete NED pipeline;
- `pretraining_born.ipynb` is the notebook which contains the code used for pre-training a Born Classifier for NED.
