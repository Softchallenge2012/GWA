@echo off
FOR %%i IN (0,1,2,3,4,5,6,7,8,9) DO (
python train.py --dataset cora --epoch 200
python train.py --dataset citeseer --epoch 200
python train.py --dataset pubmed --epoch 200
python train.py --dataset cornell --epoch 200
python train.py --dataset wisconsin --epoch 200
python train.py --dataset texas --epoch 200
python train.py --dataset chameleon --epoch 200
python train.py --dataset squirrel --epoch 200
)
pause