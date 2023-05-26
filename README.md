# Word_pair_classifier

- This is a python library for neural network that classifies correct and incorrect translation equivalents from cross-lingual word embeddings (CWEs). 
- Part of this repository are [train data](https://github.com/x-mia/Word_pair_classifier/tree/main/train_data) and [test data](https://github.com/x-mia/Word_pair_classifier/tree/main/test_data) from two CWE models trained on Estonian-Slovak language combination: [MUSE](https://github.com/facebookresearch/MUSE) and [VecMap](https://github.com/artetxem/vecmap) both trained with pre-trained [FastText](https://fasttext.cc/) (MUSE_FastText; VecMap_FastText) and [SketchEngine](https://embeddings.sketchengine.eu/) (MUSE_SketchEngine; VecMap_SketchEngine) monolingual word embeddings.
- There are also pre-trained classification models for each dataset ([pre-trained_models](https://github.com/x-mia/Word_pair_classifier/tree/main/pre-trained_models)). 

### Requirements
* [Tensorflow](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

### Train classification neural network
To train the neural network for classification, simply run:
```bash
python classify.py --precision PRECISION --train_path TRAIN_PATH --test_path TEST_PATH --plot_fig PLOT_FIG --output OUTPUT
```
Example:
```bash
python classify.py --precision 10 --train_path .\train_data\MUSE_SketchEngine.csv --test_path .\test_data\MUSE_SketchEngine.csv --plot_fig True --output my_model
```

### Predicting the new data
To predict new data using pre-trained model, simply run:
```bash
python predict.py --precision PRECISION --src_lng SRC_LNG --tgt_lng TGT_LNG --model_path MODEL_PATH --test_path TEST_PATH --output OUTPUT
```
Example:
```bash
python predict.py --precision 1 --src_lng et --tgt_lng sk --model_path my_model --test_path .\test_data\MUSE_SketchEngine.csv --output output_df.csv
```
