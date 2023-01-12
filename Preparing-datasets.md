# Instructions for preparing datasets

The followings describe steps to download and prepare all the datasets.

**WARNING: Downloading all datasets requires a long time. You can get only for what you need.**

First, create a `downloads` folder to store files and folders for datasets in this example.

    mkdir downloads

## AudioSet20K and AudioSet

As the AudioSet samples become disappearing, we create the CSV metadata files for AudioSet20K and AudioSet, while corresponding to the samples present within your environment.

To prepare audio files, please follow these steps:
- Download AudioSet samples in the .wav format.
- Convert the sampling rate of the downloaded files to your desired rate.
- Arrange the acquired files in the appropriate subfolders, namely `balanced_train_segments`, `eval_segments`, or `unbalanced_train_segments`, located under the "work/16k/as" folder. It should be noted that "16k" denotes a sampling rate of 16k Hz, and should be replaced with your rate accordingly.

To create CSV metadata files, please follow these steps:

    cd evar
    python evar/utils/make_as_metadata.py work/16k/as
    cd ..

The `make_as_metadata.py` will create `evar/metadata/as20k.csv` and `evar/metadata/as.csv`.
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

The metadata is provided by default for the reproducibility as `evar/metadata/cremad.csv`.

## ESC-50

The followings will create a folder `downloads/ESC-50-master`.

    cd downloads
    wget https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip master.zip
    cd ..

The followings will get resampled copies ready.

    python prepare_wav.py downloads/ESC-50-master work/16k/esc50 16000
    python prepare_wav.py downloads/ESC-50-master work/22k/esc50 22000
    python prepare_wav.py downloads/ESC-50-master work/32k/esc50 32000
    python prepare_wav.py downloads/ESC-50-master work/44k/esc50 44100
    python prepare_wav.py downloads/ESC-50-master work/48k/esc50 48000

The following creates metadata/esc50.csv:

    python evar/utils/make_metadata.py esc50 /data/A/ESC-50-master

## FSD50K

Download from [https://zenodo.org/record/4060432](https://zenodo.org/record/4060432) and unzip files.
If we have FSD50K files under a folder `downloads/fsd50k`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/fsd50k work/16k/fsd50k 16000
    python prepare_wav.py downloads/fsd50k work/22k/fsd50k 22000
    python prepare_wav.py downloads/fsd50k work/32k/fsd50k 32000
    python prepare_wav.py downloads/fsd50k work/44k/fsd50k 44100
    python prepare_wav.py downloads/fsd50k work/48k/fsd50k 48000

The following creates metadata/fsd50k.csv:

    python evar/utils/make_metadata.py fsd50k /data/A/fsd50k

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

    python evar/utils/make_metadata.py gtzan /data/A/gtzan

## NSynth

Download NSynth dataset and uncompress files.
If we have FSD50K files under a folder `downloads/nsynth`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/nsynth work/16k/nsynth 16000
    python prepare_wav.py downloads/nsynth work/22k/nsynth 22000
    python prepare_wav.py downloads/nsynth work/32k/nsynth 32000
    python prepare_wav.py downloads/nsynth work/44k/nsynth 44100
    python prepare_wav.py downloads/nsynth work/48k/nsynth 48000

The following creates metadata/nsynth.csv:

    python evar/utils/make_metadata.py nsynth /data/A/nsynth

## SPCV1/V2

Download Speech commands datasets and uncompress files.
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

    python evar/utils/make_metadata.py spcv2 /data/A/spcv2
    python evar/utils/make_metadata.py spcv1 /data/A/spcv1

## Pitch Audio Dataset (Surge synthesizer)

Download Surge dataset from [https://zenodo.org/record/4677097](https://zenodo.org/record/4677097) and uncompress files.
If we have files under a folder `downloads/surge`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/surge work/16k/surge 16000
    python prepare_wav.py downloads/surge work/22k/surge 22000
    python prepare_wav.py downloads/surge work/32k/surge 32000
    python prepare_wav.py downloads/surge work/44k/surge 44100
    python prepare_wav.py downloads/surge work/48k/surge 48000

The following creates metadata/surge.csv:

    python evar/utils/make_metadata.py surge downloads/surge

## UrbanSound8K

Download Surge dataset from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) and uncompress files.
If we have files under a folder `downloads/us8k`, the followings will get resampled copies ready.

    python prepare_wav.py downloads/us8k work/16k/us8k 16000
    python prepare_wav.py downloads/us8k work/22k/us8k 22000
    python prepare_wav.py downloads/us8k work/32k/us8k 32000
    python prepare_wav.py downloads/us8k work/44k/us8k 44100
    python prepare_wav.py downloads/us8k work/48k/us8k 48000

The following creates metadata/us8k.csv:

    python evar/utils/make_metadata.py us8k /data/A/us8k

## VoxCeleb1

Download Surge dataset from [The VoxCeleb1 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and uncompress files.
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

The metadata is provided by default for the reproducibility as `evar/metadata/voxforge.csv`.


(end of document)