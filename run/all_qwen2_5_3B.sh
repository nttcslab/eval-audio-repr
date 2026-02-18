NAME=QWEN2_5_3B
python 2pass_lineareval.py config/qwen2_5_3B.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5_3B.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
