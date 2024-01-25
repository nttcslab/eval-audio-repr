NAME=ATSTFrame
python 2pass_lineareval.py config/atst_frame.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/atst_frame.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
