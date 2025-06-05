base=$EVAR
weight=${base}/external/atst_base.ckpt
dim=1536
name=ATST-CLIP
GPU=0

python -m src.benchmark.processing.copd_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task copd --pretrain $name --dim $dim
python -m src.benchmark.processing.icbhi_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task icbhidisease --pretrain $name --dim $dim
python -m src.benchmark.processing.kauh_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task kauh --pretrain $name --dim $dim

python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight --label covid
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidcovid --pretrain $name --dim $dim
python -m src.benchmark.processing.coughvid_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight --label gender
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coughvidsex --pretrain $name --dim $dim

python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight --label smoker
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasmoker --pretrain $name --dim $dim --modality cough-shallow
python -m src.benchmark.processing.coswara_processing --pretrain evar:$base:config/atst.yaml:$name:weight_file=$weight --label sex
CUDA_VISIBLE_DEVICES=$GPU python -m src.benchmark.linear_eval --task coswarasex --pretrain $name --dim $dim --modality cough-shallow

