name=HTSAT
python 2pass_lineareval.py config/htsat.yaml cremad batch_size=16,name=$name
python 2pass_lineareval.py config/htsat.yaml gtzan batch_size=16,name=$name
python 2pass_lineareval.py config/htsat.yaml spcv2 batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml esc50 batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml us8k batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml vc1 batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml voxforge batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml nsynth batch_size=64,name=$name
python 2pass_lineareval.py config/htsat.yaml surge batch_size=64,name=$name
python summarize.py $name
