NAME=WavCaps
python 2pass_lineareval.py config/wavcaps.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/wavcaps.yaml surge batch_size=64,name=$NAME

python zeroshot.py config/wavcaps.yaml cremad batch_size=16,name=$NAME
python zeroshot.py config/wavcaps.yaml gtzan batch_size=16,name=$NAME
python zeroshot.py config/wavcaps.yaml nsynth batch_size=64,name=$NAME
python zeroshot.py config/wavcaps.yaml esc50 batch_size=64,name=$NAME
python zeroshot.py config/wavcaps.yaml us8k batch_size=64,name=$NAME

python summarize.py $NAME
