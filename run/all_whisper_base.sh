NAME=Whisper_base
python 2pass_lineareval.py config/whisper.yaml cremad batch_size=16,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml gtzan batch_size=16,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml spcv2 batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml esc50 batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml us8k batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml vc1 batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml voxforge batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml nsynth batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python 2pass_lineareval.py config/whisper.yaml surge batch_size=64,pretrained_model=openai/whisper-base,name=$NAME
python summarize.py $NAME
