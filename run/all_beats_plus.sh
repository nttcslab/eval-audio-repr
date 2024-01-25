NAME=BEATsPlus
python 2pass_lineareval.py config/beats_plus.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/beats_plus.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
