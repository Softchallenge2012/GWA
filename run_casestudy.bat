@echo off
FOR %%i IN (100,500,1000,1500,2000,2500,3000,3500,4000) DO (
python casestudy.py --scale 0.2 --num_nodes %%i --nx_type heterophily
python casestudy.py --scale 0.5 --num_nodes %%i --nx_type heterophily
python casestudy.py --scale 0.7 --num_nodes %%i --nx_type heterophily
)
pause