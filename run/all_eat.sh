NAME=EAT_Base
python 2pass_lineareval.py config/eat.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/eat.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/eat.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/eat.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
