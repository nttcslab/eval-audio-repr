NAME=CED
python 2pass_lineareval.py config/ced.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/ced.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/ced.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/ced.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
