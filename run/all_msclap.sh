NAME=MSCLAP
python 2pass_lineareval.py config/msclap.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/msclap.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/msclap.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml surge batch_size=64,name=$NAME

python zeroshot.py config/msclap.yaml cremad batch_size=16,name=$NAME
python zeroshot.py config/msclap.yaml gtzan batch_size=16,name=$NAME
python zeroshot.py config/msclap.yaml nsynth batch_size=64,name=$NAME
python zeroshot.py config/msclap.yaml esc50 batch_size=64,name=$NAME
python zeroshot.py config/msclap.yaml us8k batch_size=64,name=$NAME

python summarize.py $NAME
