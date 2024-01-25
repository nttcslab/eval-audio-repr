NAME=ATST
python 2pass_lineareval.py config/atst.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/atst.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/atst.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
