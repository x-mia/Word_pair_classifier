# Word_pair_classifier

This is a Python library for neural network that classifies correct and incorrect translation equivalents from cross-lingual word embeddings (CWEs). Part of this repository are [testing and training data](https://github.com/x-mia/Word_pair_classifier/tree/main/Data) and [models](https://github.com/x-mia/Word_pair_classifier/tree/main/Models) for five language combinations (et-sk, en-fr, en-cs, en-fi, and en-ko). These models are pre-trained for two CWE models: [MUSE](https://github.com/facebookresearch/MUSE) and [VecMap](https://github.com/artetxem/vecmap) both trained with [FastText](https://fasttext.cc/) and [SketchEngine](https://embeddings.sketchengine.eu/) (et-sk only) monolingual word embeddings.

### Requirements
* [Tensorflow](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

### Train classification neural network
To train the neural network for classification, simply run:
```bash
python classify.py --train_path TRAIN_PATH --test_path TEST_PATH --plot_fig PLOT_FIG --output OUTPUT
```
Example:
```bash
python classify.py --train_path .\train_data\MUSE_SketchEngine.csv --test_path .\test_data\MUSE_SketchEngine.csv --plot_fig True --output my_model
```

### Predicting the new data
To predict new data using an already trained model, simply run:
```bash
python predict.py --src_lng SRC_LNG --tgt_lng TGT_LNG --model_path MODEL_PATH --test_path TEST_PATH --output OUTPUT
```
Example:
```bash
python predict.py --src_lng et --tgt_lng sk --model_path my_model --test_path .\test_data\MUSE_SketchEngine.csv --output output_df.csv
```

### Related work
* [M. Artetxe, G. Labaka, E. Agirre - *Learning principled bilingual mappings of word embeddings while preserving monolingual invariance*, 2016](https://aclanthology.org/D16-1250/)
* [A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou - *Word Translation Without Parallel Data*, 2017](https://arxiv.org/pdf/1710.04087.pdf)
* [E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov - *Learning Word Vectors for 157 Languages*, 2018](https://arxiv.org/abs/1802.06893)
* [O. Herman - *Precomputed Word Embeddings for 15+ Languages*, 2021](https://www.sketchengine.eu/wp-content/uploads/2021-Precomputed-Word-Embeddings.pdf)
