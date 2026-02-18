NAME=OWSM_3.1_ebf
python 2pass_lineareval.py config/owsm.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/owsm.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/owsm.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/owsm.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
