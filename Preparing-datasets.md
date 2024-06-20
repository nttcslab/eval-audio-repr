# Instructions for preparing datasets

The followings describe steps to download and prepare all the datasets.

**WARNING: Downloading all datasets requires a long time. You can get only for what you need.**

First, create a `downloads` folder to store files and folders for datasets in this example.

    mkdir downloads

## AudioSet20K and AudioSet (2M)

As the AudioSet samples become disappearing, we create the CSV metadata files for AudioSet20K and AudioSet, while corresponding to the samples present within your environment.

To prepare audio files, please follow these steps:
- Download AudioSet samples in the .wav format.
- Convert the sampling rate of the downloaded files to your desired rate.
- Arrange the acquired files in the appropriate subfolders, namely `balanced_train_segments`, `eval_segments`, or `unbalanced_train_segments`, located under the "work/16k/as" folder. It should be noted that "16k" denotes a sampling rate of 16k Hz, and should be replaced with your rate accordingly.

To create CSV metadata files, please follow these steps:

    cd evar
    python evar/utils/make_as_metadata.py work/16k/as
    cd ..

The `make_as_metadata.py` will create `evar/metadata/as20k.csv`, `evar/metadata/as.csv`, and `evar/metadata/weight_as.csv`.
Please note that "16k" denotes a sampling rate of 16k Hz and should be replaced with your rate accordingly.

## CREMA-D

The followings will download to `downloads/cremad`.

    python evar/utils/download_cremad.py downloads/cremad

The followings will get resampled copies ready.

    python prepare_wav.py downloads/cremad work/16k/cremad 16000
    python prepare_wav.py downloads/cremad work/22k/cremad 22000
    python prepare_wav.py downloads/cremad work/32k/cremad 32000
    python prepare_wav.py downloads/cremad work/44k/cremad 44100
    python prepare_wav.py downloads/cremad work/48k/cremad 48000

We provide the metadata as `evar/metadata/cremad.csv` by default for reproducibility purposes.

## ESC-50

The followings will create a folder `downloads/esc50`.

    cd downloads
    wget https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip master.zip
    mv ESC-50-master esc50
    cd ..

The followings will get resampled copies ready.

    python prepare_wav.py downloads/esc50 work/16k/esc50 16000
    python prepare_wav.py downloads/esc50 work/22k/esc50 22000
    python prepare_wav.py downloads/esc50 work/32k/esc50 32000
    python prepare_wav.py downloads/esc50 work/44k/esc50 44100
    python prepare_wav.py downloads/esc50 work/48k/esc50 48000

The following creates metadata/esc50.csv:

    python evar/utils/make_metadata.py esc50 /your/esc50

## FSD50K

Download from [https://zenodo.org/record/4060432](https://zenodo.org/record/4060432) and unzip files.
If we have FSD50K files under a folder `downloads/fsd50k`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/fsd50k work/16k/fsd50k 16000
    python prepare_wav.py downloads/fsd50k work/22k/fsd50k 22000
    python prepare_wav.py downloads/fsd50k work/32k/fsd50k 32000
    python prepare_wav.py downloads/fsd50k work/44k/fsd50k 44100
    python prepare_wav.py downloads/fsd50k work/48k/fsd50k 48000

The following creates metadata/fsd50k.csv:

    python evar/utils/make_metadata.py fsd50k /your/fsd50k

## GTZAN

The followings will create a folder `downloads/gtzan`.

    cd downloads
    wget http://opihi.cs.uvic.ca/sound/genres.tar.gz 
    tar xf genres.tar.gz
    mv genres gtzan
    cd ..

The followings will get resampled copies ready.

    python prepare_wav.py downloads/gtzan work/16k/gtzan 16000
    python prepare_wav.py downloads/gtzan work/22k/gtzan 22000
    python prepare_wav.py downloads/gtzan work/32k/gtzan 32000
    python prepare_wav.py downloads/gtzan work/44k/gtzan 44100
    python prepare_wav.py downloads/gtzan work/48k/gtzan 48000

The following creates metadata/gtzan.csv:

    python evar/utils/make_metadata.py gtzan /your/gtzan

## NSynth

Download NSynth dataset and uncompress files.

    mkdir nsynth
    (cd nsynth && tar xf /path/to/nsynth-test.jsonwav.tar.gz)
    (cd nsynth && tar xf /path/to/nsynth-valid.jsonwav.tar.gz)
    (cd nsynth && tar xf /path/to/nsynth-train.jsonwav.tar.gz)

If we have NSynth files under a folder `downloads/nsynth`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/nsynth work/16k/nsynth 16000
    python prepare_wav.py downloads/nsynth work/22k/nsynth 22000
    python prepare_wav.py downloads/nsynth work/32k/nsynth 32000
    python prepare_wav.py downloads/nsynth work/44k/nsynth 44100
    python prepare_wav.py downloads/nsynth work/48k/nsynth 48000

The following creates metadata/nsynth.csv:

    python evar/utils/make_metadata.py nsynth /your/nsynth

## SPCV1/V2

Download Speech commands datasets and uncompress files.

    mkdir spcv2
    (cd spcv2 && tar xf /path/to/speech_commands_v0.02.tar.gz)


If we have files under folder `downloads/spcv1` and  `downloads/spcv2`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/spcv1 work/16k/spcv1 16000
    python prepare_wav.py downloads/spcv1 work/22k/spcv1 22000
    python prepare_wav.py downloads/spcv1 work/32k/spcv1 32000
    python prepare_wav.py downloads/spcv1 work/44k/spcv1 44100
    python prepare_wav.py downloads/spcv1 work/48k/spcv1 48000

    python prepare_wav.py downloads/spcv2 work/16k/spcv2 16000
    python prepare_wav.py downloads/spcv2 work/22k/spcv2 22000
    python prepare_wav.py downloads/spcv2 work/32k/spcv2 32000
    python prepare_wav.py downloads/spcv2 work/44k/spcv2 44100
    python prepare_wav.py downloads/spcv2 work/48k/spcv2 48000

The following creates metadata/spcvX.csv:

    python evar/utils/make_metadata.py spcv2 /your/spcv2
    python evar/utils/make_metadata.py spcv1 /your/spcv1

## Pitch Audio Dataset (Surge synthesizer)

Download Surge dataset from [https://zenodo.org/record/4677097](https://zenodo.org/record/4677097) and uncompress files.
If we have files under a folder `downloads/surge`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/surge work/16k/surge 16000 --suffix .ogg
    python prepare_wav.py downloads/surge work/22k/surge 22000 --suffix .ogg
    python prepare_wav.py downloads/surge work/32k/surge 32000 --suffix .ogg
    python prepare_wav.py downloads/surge work/44k/surge 44100 --suffix .ogg
    python prepare_wav.py downloads/surge work/48k/surge 48000 --suffix .ogg

The following creates metadata/surge.csv:

    python evar/utils/make_metadata.py surge downloads/surge

## UrbanSound8K

Download Surge dataset from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) and uncompress files.

    tar xf /path/to/UrbanSound8K.tgz
    mv UrbanSound8K us8k

If we have files under a folder `downloads/us8k`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/us8k work/16k/us8k 16000
    python prepare_wav.py downloads/us8k work/22k/us8k 22000
    python prepare_wav.py downloads/us8k work/32k/us8k 32000
    python prepare_wav.py downloads/us8k work/44k/us8k 44100
    python prepare_wav.py downloads/us8k work/48k/us8k 48000

The following creates metadata/us8k.csv:

    python evar/utils/make_metadata.py us8k /your/us8k

## VoxCeleb1

Download Surge dataset from [The VoxCeleb1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and uncompress files.

    tar xf /path/to/VoxCeleb1/VoxCeleb1.tgz
    mv VoxCeleb1\ Dataset vc1
    (cd vc1 && unzip /path/to/VoxCeleb1/vox1_test_wav.zip)

If we have files under a folder `downloads/vc1`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/vc1 work/16k/vc1 16000
    python prepare_wav.py downloads/vc1 work/22k/vc1 22000
    python prepare_wav.py downloads/vc1 work/32k/vc1 32000
    python prepare_wav.py downloads/vc1 work/44k/vc1 44100
    python prepare_wav.py downloads/vc1 work/48k/vc1 48000

The following creates metadata/vc1.csv:

    python evar/utils/make_metadata.py vc1

## VoxForge

The followings will download to `downloads/voxforge`.

    python evar/utils/download_voxforge.py downloads/voxforge

The followings will get resampled copies ready.

    python prepare_wav.py downloads/voxforge work/16k/voxforge 16000
    python prepare_wav.py downloads/voxforge work/22k/voxforge 22000
    python prepare_wav.py downloads/voxforge work/32k/voxforge 32000
    python prepare_wav.py downloads/voxforge work/44k/voxforge 44100
    python prepare_wav.py downloads/voxforge work/48k/voxforge 48000

We provide the metadata as `evar/metadata/voxforge.csv` by default for reproducibility purposes.

## Audio Captioning (Audio to Text Retrieval) Datasets

### AudioCaps/ML-AudioCaps

For downloading the audio data, refer to the instruction: https://github.com/XinhaoMei/ACT?tab=readme-ov-file#set-up-dataset

Place the files under `work/original` folder:

    work/original/audiocaps
      /train
      /test
      /val
        *.wav files

The following will make the metadata for AudioCaps.

    python evar/utils/make_metadata.py audiocaps /path/to/ml-audiocaps/en

The following will make the metadata for the Japanese captions of ML-AudioCaps.

    python evar/utils/make_metadata.py ja_audiocaps /path/to/ml-audiocaps/ja

### Clotho

Unpack the 7z archives and place the files under `work/original` folder:

    work/original/clotho
      /evaluation
      /validation
        *.wav files

The following will make the metadata for Clotho. The `/path/to/clotho` should have *.csv files.

    python evar/utils/make_metadata.py clotho /path/to/clotho

## Example command lines

Once all the original datasets are ready in the `$FROM` folder, the following command lines will convert the sampling rate of files and store them in the `$TO` folder.

### For 16 kHz

```sh
export FROM=/hdd/datasets/evar_original
export TO=/lab/evar_work_new/16k
export SR=16000
python prepare_wav.py $FROM/cremad $TO/cremad $SR
python prepare_wav.py $FROM/esc50 $TO/esc50 $SR
python prepare_wav.py $FROM/gtzan $TO/gtzan $SR
python prepare_wav.py $FROM/nsynth $TO/nsynth $SR
python prepare_wav.py $FROM/spcv1 $TO/spcv1 $SR
python prepare_wav.py $FROM/spcv2 $TO/spcv2 $SR
python prepare_wav.py $FROM/surge $TO/surge $SR --suffix .ogg
python prepare_wav.py $FROM/us8k $TO/us8k $SR
python prepare_wav.py $FROM/vc1 $TO/vc1 $SR
python prepare_wav.py $FROM/voxforge $TO/voxforge $SR
python prepare_wav.py $FROM/fsd50k $TO/fsd50k $SR
```

### For 32 kHz

```sh
export FROM=/hdd/datasets/evar_original
export TO=/biglab/evar_work_new/32k
export SR=32000
python prepare_wav.py $FROM/cremad $TO/cremad $SR
python prepare_wav.py $FROM/esc50 $TO/esc50 $SR
python prepare_wav.py $FROM/gtzan $TO/gtzan $SR
python prepare_wav.py $FROM/nsynth $TO/nsynth $SR
python prepare_wav.py $FROM/spcv1 $TO/spcv1 $SR
python prepare_wav.py $FROM/spcv2 $TO/spcv2 $SR
python prepare_wav.py $FROM/surge $TO/surge $SR --suffix .ogg
python prepare_wav.py $FROM/us8k $TO/us8k $SR
python prepare_wav.py $FROM/vc1 $TO/vc1 $SR
python prepare_wav.py $FROM/voxforge $TO/voxforge $SR
python prepare_wav.py $FROM/fsd50k $TO/fsd50k $SR
```

## Zero-shot evaluation data

The zero-shot evaluation uses the original data files from the tasks. Store (or make symbolic links) the folders of the original data under a folder `work/original` as follows:

    work/original/AudioSet
    work/original/ESC-50-master
    work/original/UrbanSound8K
    work/original/cremad
    work/original/fsd50k
    work/original/gtzan
    work/original/nsynth

For AudioSet, download the class definition CSV.

    wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv


(end of document)