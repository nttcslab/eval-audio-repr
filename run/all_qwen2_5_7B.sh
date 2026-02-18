NAME=QWEN2_5_7B
python 2pass_lineareval.py config/qwen2_5.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
