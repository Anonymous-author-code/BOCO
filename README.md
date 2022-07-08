# BOCO

This is the implementation of paper "Bilevel Optimization for Learning Transportation Network Combinatorial Optimization".



## Environment set up

This code is developed and tested on Ubuntu 16.04 with Python 3.6.9, Pytorch 1.7.1, CUDA 10.1.

Install required pacakges:
```shell
export TORCH=1.7.0
export CUDA=cu101
pip install torch==1.7.1+${CUDA} torchvision==0.8.2+${CUDA} torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install --no-index --upgrade torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --upgrade torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index --upgrade torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install --upgrade torch-geometric
pip install tensorboard
pip install networkx==2.2
pip install ortools
pip install texttable
pip install tsplib95
pip install cython
```



Install [LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3/) which is required by the HCP experiment:
```
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
tar xvfz LKH-3.0.6.tgz
cd LKH-3.0.6
make
```
And you should find an executable at ``./LKH-3.0.6/LKH``, which will be called by our code.

## Run Experiments
We provide the implementation of AC-BOCO and the single-level RL baseline AC-Single used in our paper. To run evaluation from a pretrained model, replace ``train`` by ``eval`` in the following commands.



### TSP
AC-BOCO:
```
python ged_AC_BOCO_train.py --config AC_BOCO_ged.yaml
```
AC-Single:
```
python ged_AC_single_train.py --config AC_single_ged.yaml
```
To test different problem sizes, please modify this entry in the yaml file: ``dataset: AIDS-20-30`` (to reproduce the results in our paper, please set AIDS-20-30/AIDS-30-50/AIDS-50-100)

### CVRP 
AC-BOCO:
```
python hcp_AC-BOCO_train.py --config AC-BOCO_hcp.yaml
```
AC-Single:
```
python hcp_AC_single_train.py --config AC_single_hcp.yaml
```

### Some Remarks
The yaml configs are set for the smallest sized problems by default. For AC-Single, you may need to adjust the ``max_timesteps`` config for larger-sized problems to ensures that the RL agent can predict a valid solution.

## Pretrained models
We provide pretrained models for PPO-BiHyb on these three problems, which are stored in ``./pretrained``. To use your own parameters, please set the ``test_model_weight`` configuration in the yaml file.



We would like to give credits to the following online resources and thank their great work:
* [TPC](http://www.tpc.org/tpch/) (for our DAG scheduling dataset)
* [GEDLIB](https://github.com/dbblumenthal/gedlib) and [U.S. National Institutes of Health](https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data) (for our GED dataset)
* [FHCP challenge](https://sites.flinders.edu.au/flinders-hamiltonian-cycle-project/fhcp-challenge-set/) (for our HCP dataset)
* [LKH-3](http://webhotel4.ruc.dk/~keld/research/LKH-3/) (the powerful HCP/TSP heuristic)
