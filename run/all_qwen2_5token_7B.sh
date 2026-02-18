NAME=QWEN2_5token_7B
python 2pass_lineareval.py config/qwen2_5token_7B.yaml cremad batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml gtzan batch_size=16,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml spcv2 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml esc50 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml us8k batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml vc1 batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml voxforge batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml nsynth batch_size=64,name=$NAME
python 2pass_lineareval.py config/qwen2_5token_7B.yaml surge batch_size=64,name=$NAME
python summarize.py $NAME
