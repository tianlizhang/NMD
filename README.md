# NMD
This repository contains the code for NMD: Disentangling Node Metric Factor For Temporal Link Prediction.

## Dataset
* [UCI](http://konect.uni-koblenz.de/networks/opsahl-ucsocial): uc_irvine. Download this file to raw_data/
* [BCA](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html): BitCoin Alpha.  Download this file to raw_data/bitcoin
* [BCO](http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html):  BitCoin OTC. Download this file to raw_data/bitcoin
* [AS](http://snap.stanford.edu/data/as-733.html): Autonomous Systems. Download this file to raw_data
* [APS](https://journals.aps.org/datasets): Pyhsical Dataset. Download this file to raw_data/aps
* [DBLP](https://www.aminer.org/citation):  DBLP-Citation-network V13:  5,354,309 papers and 48,227,950 citation relationships (2021-05-14). Download this file to raw_data/dblp


## Requirments
* Python 3.6
* pyyaml-6.0
* blessed-1.19.1 gpustat-1.0.0 nvidia-ml-py-11.495.46 psutil-5.9.4
* pandas-1.5.3 python-dateutil-2.8.2 pytz-2022.7.1
* IPython-8.8.0 
* tqdm-4.64.1
* contourpy-1.0.7 cycler-0.11.0 fonttools-4.38.0 kiwisolver-1.4.4 matplotlib-3.6.3 packaging-23.0 pyparsing-3.0.9
* joblib-1.2.0 scikit-learn-1.2.0 scipy-1.10.0 threadpoolctl-3.1.0
* PyTorch 1.0 or higher: https://pytorch.org/get-started/locally/
* Dgl 0.6.1

```bash
pip install pyyaml
pip intall IPython
pip install pandas
pip install gpustat
pip install matplotlib
pip install -U scikit-learn
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
pip install dgl-cu102==0.6.1
```

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install dgl-cu102==0.6.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
```


### Preprocess DBLP and APS
* Run preprocess/ipynb/dblp.ipynb
* Run preprocess/ipynb/aps.ipynb
* Run preprocess/ipynb/k_core.py



## Run
```bash
cd src
python run_exp.py --data uci --model d2v
```

## Experiment:
### Result:
```bash
bash 01_run.sh
```

### Fusion mode:
```bash
bash 02_fusion_mode.sh
```

### Ablation study:
```bash
bash 03_ablation.sh
```

### Parameter Sensitivity
```bash
bash 04_parameter.sh
```

### RQ4: Effect of TIM
* Run commands to save models in ckpt file:
```bash
bash 01_run.sh
```
* If you select the d2v model of the 0 epoch on the dataset BCA, then run the following command to generate the degree-mrr values:
```bash
cd show
python cites_mrr.py --data bca --model d2v --epoch 0
or bash cmd.sh
```
* Copy the values, open the ipynb (file)[show/grpup.ipynb], paste the values and draw the degree-mrr figure.

### Performance on TIP task
```bash
bash TIP_cmd.sh
```

### Shared Structrual Encoder
```bash
bash 05_SSE.sh
```

### Symmetrical Fusion
```bash
bash 06_SF.sh
```