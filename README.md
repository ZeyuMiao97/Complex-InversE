# Complex-InversE
a Novel Approach for Knowledge Graph Link Prediction in Complex Space
# Summary
This software can be used to reproduce the results in our "Complex-InversE: Improving Bilinear Knowledge Graph
Embeddings by Mapping into Complex Space" paper. It can be also used to learn `Complex-InversE` models for other datasets. The software can be also used as a framework to implement new tensor factorization models (implementations for `TransE` , `ComplEx` and `SimplE` are included as two examples).
# Dependencies
* `Python` version 2.7
* `Numpy` version 1.13.1
* `Tensorflow` version 1.1.0
# Usage
To run a model `M` on a dataset `D`, do the following steps:
* `cd` to the directory where `main.py` is  
* Run `python main.py -m M -d D`  
  
Examples (commands start after $):  
  
`$ python main.py -m ED-SimplE -d wn18`  
`$ python main.py -m SimplE_avg -d fb15k`  
`$ python main.py -m ComplEx -d wn18`  
  
Running a model `M` on a dataset `D` will save the embeddings in a folder with the following address:  
  
`$ <Current Directory>/M_weights/D/`  
  
As an example, running the `Complex-InversE` model on wn18 will save the embeddings in the following folder:  
  
`$ <Current Directory>/Complex-InversE_weights/wn18/`  
 
# Datasets
The following datasets are available in this software：
* `FB15k`
* `WN18`  

If you want to run Complex-InversE on dataset FB15k-237 or WN18RR, please visit [Complex-InversE-torch](https://github.com/ZeyuMiao97/Complex-InversE-torch)
