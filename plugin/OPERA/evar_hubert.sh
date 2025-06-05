base=$EVAR
dim=768
GPU=0
# filename=$(basename $weight)
# "$(basename "$(dirname "$weight")")_${filename%.*}"

for i in 1 2 3 4 5 6 7 8 9 10 11 12; do

name="HuBERT_$i"
python -m src.benchmark.processing.copd_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i]
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task copd --pretrain $name --dim $dim
python -m src.benchmark.processing.icbhi_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i]
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task icbhidisease --pretrain $name --dim $dim
python -m src.benchmark.processing.kauh_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i]
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task kauh --pretrain $name --dim $dim

python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i] --label covid
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidcovid --pretrain $name --dim $dim
python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i] --label gender
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidsex --pretrain $name --dim $dim

python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i] --label smoker
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasmoker --pretrain $name --dim $dim --modality cough-shallow
python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/hubert.yaml:$name:output_layers=[$i] --label sex
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasex --pretrain $name --dim $dim --modality cough-shallow

done
