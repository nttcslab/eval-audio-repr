#
NAME=$1
WEIGHT=$2
SEED=42
ITER=5
FEATURES=768
FEAT_NAME=$NAME

if [ $# -gt 2 ]; then
    SEED=$3
    echo "Seed = $SEED."
fi
if [ $# -gt 3 ]; then
    ITER=$4
    echo "Number of iteration = $ITER."
fi
if [ $# -gt 4 ]; then
    FEAT_NAME=$5
    echo "Feature name = $FEAT_NAME."
fi
if [ $# -gt 5 ]; then
    FEATURES=$6
    echo "Num_features = $FEATURES."
fi

OPTION="dataset.pre_extract.output_dir=outputs/feat/evar_"$FEAT_NAME"_feats,,dataset.input_dir=outputs/feat/evar_"$FEAT_NAME"_feats,,dataset.pre_extract.feature_extractor.pretrain.evar_config=$EVAR/config/$NAME.yaml,,dataset.pre_extract.feature_extractor.pretrain.weight=$WEIGHT,,dataset.pre_extract.feature_extractor.pretrain.num_features=$FEATURES"

#GS
TASKS="EMO GTZAN MTT"
for task in $TASKS; do
  python . extract -c configs/evar/$task.yaml -o $OPTION",,trainer.seed=$SEED"
  for i in $(seq $ITER); do
    python . probe -c configs/evar/$task.yaml -o $OPTION",,trainer.seed=$SEED"
    SEED=$((SEED + 1))
  done
done

python . extract -c configs/evar/VocalSetS.yaml -o $OPTION
TASKS="VocalSetS VocalSetT"
for task in $TASKS; do
  for i in $(seq $ITER); do
    python . probe -c configs/evar/$task.yaml -o $OPTION",,trainer.seed=$SEED"
    SEED=$((SEED + 1))
  done
done

python . extract -c configs/evar/NSynthI.yaml -o $OPTION
TASKS="NSynthI NSynthP"
for task in $TASKS; do
  for i in $(seq $ITER); do
    python . probe -c configs/evar/$task.yaml -o $OPTION",,trainer.seed=$SEED"
    SEED=$((SEED + 1))
  done
done

python . extract -c configs/evar/MTGGenre.yaml -o $OPTION
TASKS="MTGGenre MTGInstrument MTGMood MTGTop50"
for task in $TASKS; do
  for i in $(seq $ITER); do
    python . probe -c configs/evar/$task.yaml -o $OPTION",,trainer.seed=$SEED"
    SEED=$((SEED + 1))
  done
done
