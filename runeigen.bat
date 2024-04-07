@echo off
FOR %%i IN (0,1,2,3,4,5,6,7,8,9) DO (
python train_eigen.py --dataset cornell --epoch 200
python train_eigen.py --dataset wisconsin --epoch 200
python train_eigen.py --dataset texas --epoch 200
)
pause