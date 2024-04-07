
for i in (100,200,300,400) do 
	python casestudy.py --num_nodes i --scale 0.2 --nx_type homophily
	python casestudy.py --num_nodes i --scale 0.2 --nx_type random
	python casestudy.py --num_nodes i --scale 0.2 --nx_type heterophily
