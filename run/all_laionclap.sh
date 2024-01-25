NAME=LAIONCLAP
python 2pass_lineareval.py config/laionclap.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/laionclap.yaml surge batch_size=64,name=$NAME

python zeroshot.py config/laionclap.yaml cremad batch_size=16,name=$NAME
python zeroshot.py config/laionclap.yaml gtzan batch_size=16,name=$NAME
python zeroshot.py config/laionclap.yaml nsynth batch_size=64,name=$NAME
python zeroshot.py config/laionclap.yaml esc50 batch_size=64,name=$NAME
python zeroshot.py config/laionclap.yaml us8k batch_size=64,name=$NAME

python summarize.py $NAME
