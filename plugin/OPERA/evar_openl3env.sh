base=$EVAR
dim=6144
name=OpenL3
GPU=0
# filename=$(basename $weight)
# "$(basename "$(dirname "$weight")")_${filename%.*}"

python -m src.benchmark.processing.copd_processing --pretrain evar:$base:config/openl3env.yaml:$name
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task copd --pretrain $name --dim $dim
python -m src.benchmark.processing.icbhi_processing --pretrain evar:$base:config/openl3env.yaml:$name
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task icbhidisease --pretrain $name --dim $dim
python -m src.benchmark.processing.kauh_processing --pretrain evar:$base:config/openl3env.yaml:$name
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task kauh --pretrain $name --dim $dim

python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/openl3env.yaml:$name --label covid
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidcovid --pretrain $name --dim $dim
python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/openl3env.yaml:$name --label gender
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidsex --pretrain $name --dim $dim

python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/openl3env.yaml:$name --label smoker
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasmoker --pretrain $name --dim $dim --modality cough-shallow
python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/openl3env.yaml:$name --label sex
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasex --pretrain $name --dim $dim --modality cough-shallow

