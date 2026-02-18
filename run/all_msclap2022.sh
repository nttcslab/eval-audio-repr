NAME=CLAP2022
python 2pass_lineareval.py config/msclap.yaml cremad weight_file=2022,batch_size=16,name=$NAME
python 2pass_lineareval.py config/msclap.yaml gtzan weight_file=2022,batch_size=16,name=$NAME
python 2pass_lineareval.py config/msclap.yaml spcv2 weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml esc50 weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml us8k weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml vc1 weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml voxforge weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml nsynth weight_file=2022,batch_size=64,name=$NAME
python 2pass_lineareval.py config/msclap.yaml surge weight_file=2022,batch_size=64,name=$NAME

python zeroshot.py config/msclap.yaml cremad weight_file=2022,batch_size=16,name=$NAME
python zeroshot.py config/msclap.yaml gtzan weight_file=2022,batch_size=16,name=$NAME
python zeroshot.py config/msclap.yaml nsynth weight_file=2022,batch_size=64,name=$NAME
python zeroshot.py config/msclap.yaml esc50 weight_file=2022,batch_size=64,name=$NAME
python zeroshot.py config/msclap.yaml us8k weight_file=2022,batch_size=64,name=$NAME

python summarize.py $NAME
