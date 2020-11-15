# Span-GCN
Code of the paper [Graph Convolutions over Constituent Trees for Syntax-Aware Semantic Role Labeling](https://www.aclweb.org/anthology/2020.emnlp-main.322.pdf), (EMNLP 2020).

Requires python 3.5, pytorch 1.0.0, pytorch_transformers 1.2.0, [GloVe embeddings](https://pytorch.org/get-started/locally/) and [FrameNet 1.5 data](https://drive.google.com/file/d/15n3M4AmURGdGqnNAjn352buUTV5S-fVI/view?usp=sharing).
# PropBank
For training SpanGCN, you must run run_srl.py
````
python run_srl.py 
    --outputdir .
    --outputmodelname conll2005-srl
    --n_epochs 50
    --batch_size 32
    --corpus 2005
    --train-file data/conll2005/train-set-pred_syn_dep_conll
    --dev-file data/conll2005/dev-set-pred_syn_dep_conll
    --glove-path data/glove/glove.6B.100d.txt
    --emb-dim 100
    --use-syntax 1
    --embedding-layer-norm 1
    --enc-lstm-dim 300
    --n-layers 4
    --n-layers-top 2
    --word-drop 0.1
    --bilinear-dropout 0.1
    --gcn-dropout 0.2
    --emb-dropout 0.1
    --gpu-id -1
    --use-elmo 0
    --use-bert 0

````
The model accepts standard CoNLL-2005 and CoNLL-2012 format.

To run evaluation on the test set you must use the same hyperparameters of the model you want to load
````
python run_inference.py 
    --outputdir .
    --modelname conll2005-srl
    --batch_size 32
    --corpus 2005
    --train-file data/conll2005/train-set-pred_syn_dep_conll
    --dev-file data/conll2005/dev-set-pred_syn_dep_conll
    --test-file data/conll2005/test-set-pred_syn_dep_conll
    --glove-path data/glove/glove.6B.100d.txt
    --emb-dim 100
    --use-syntax 1
    --embedding-layer-norm 1
    --enc-lstm-dim 300
    --n-layers 4
    --n-layers-top 2
    --word-drop 0.1
    --bilinear-dropout 0.1
    --gcn-dropout 0.2
    --emb-dropout 0.1
    --gpu-id -1
    --use-elmo 0
    --use-bert 0
````

# FrameNet
Preprocessed training data for FrameNet are in ````data/framenet````

For training you must run run_srl_framenet.py

````
python run_srl_framenet.py 
    --outputdir .
    --outputmodelname framenet-srl
    --n_epochs 30
    --batch_size 8
    --train-file data/framenet/FN_development_conll.txt_pred_syn_pos
    --dev-file data/framenet/FN_development_conll.txt_pred_syn_pos
    --glove-path data/glove/glove.6B.100d.txt
    --ontology-path data/framenet/fndata-1.5/
    --emb-dim 100
    --use-syntax 1
    --embedding-layer-norm 1
    --enc-lstm-dim 200
    --n-layers 4
    --n-layers-top 2
    --word-drop 0.1
    --bilinear-dropout 0.1
    --gcn-dropout 0.1
    --emb-dropout 0.1
    --gpu-id -1
````

To run evaluation on the test set you must use the same hyperparameters of the model you want to load
````
python run_inference_framenet.py 
    --dir .
    --modelname framenet-srl
    --batch_size 8
    --train-file data/framenet/FN_development_conll.txt_pred_syn_pos
    --dev-file data/framenet/FN_development_conll.txt_pred_syn_pos
    --test-file data/framenet/FN_development_conll.txt_pred_syn_pos
    --glove-path data/glove/glove.6B.100d.txt
    --ontology-path data/framenet/fndata-1.5/
    --emb-dim 100
    --use-syntax 1
    --embedding-layer-norm 1
    --enc-lstm-dim 200
    --n-layers 4
    --n-layers-top 2
    --word-drop 0.1
    --bilinear-dropout 0.1
    --gcn-dropout 0.1
    --emb-dropout 0.1
    --gpu-id -1
````