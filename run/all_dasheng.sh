NAME=Dasheng_1_2B
python 2pass_lineareval.py config/dasheng.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/dasheng.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
