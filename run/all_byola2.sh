python 2pass_lineareval.py config/byola2.yaml esc50  --lr=0.001
python 2pass_lineareval.py config/byola2.yaml us8k  --lr=0.00003
python 2pass_lineareval.py config/byola2.yaml spcv2  --lr=0.00003
python 2pass_lineareval.py config/byola2.yaml nsynth  --lr=0.001
python 2pass_lineareval.py config/byola2.yaml vc1  --lr=0.00004
python 2pass_lineareval.py config/byola2.yaml voxforge  --lr=0.0001
python 2pass_lineareval.py config/byola2.yaml cremad 
python 2pass_lineareval.py config/byola2.yaml surge  --lr=0.00003
python 2pass_lineareval.py config/byola2.yaml gtzan batch_size=64 --lr=0.001
python summarize.py external/byol_a/v2/AudioNTT2022-BYOLA-64x96d2048.pth
