NAME=BEATs
python 2pass_lineareval.py config/beats.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/beats.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/beats.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
