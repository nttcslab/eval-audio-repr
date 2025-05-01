# Instructions for preparing models

The followings are command lines to prepare models.

**Note: you can setup only the models you need.**

## AST

    cd external/
    git clone https://github.com/YuanGongND/ast.git
    patch -p1 < ast_models.patch
    pip install wget
    cd ..

## ATST & ATST-Frame

In addition to the following steps, please download the ATST-Frame checkpoint as `external/atstframe_base.ckpt` from https://github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atstframe.

    (cd external && git clone https://github.com/Audio-WestlakeU/audiossl.git)
    (cd external && wget https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/base.ckpt -O atst_base.ckpt)
    pip install pytorch_lightning fairseq

## BEATs

In addition to the following steps, please download the BEATs_iter3 and BEATs_iter3_plus checkpoints as `external/BEATs_iter3.pt` and `external/BEATs_iter3_plus_AS2M.pt` from https://github.com/microsoft/unilm/tree/master/beats.

    (cd external && git clone https://github.com/microsoft/unilm.git)

## BYOL-A (IJCNN2021) & BYOL-A v2 (TASLP2023)

    cd external/
    git clone https://github.com/nttcslab/byol-a.git
    mv byol-a byol_a
    cd ..

## CED

    (cd external && git clone https://github.com/jimbozhang/hf_transformers_custom_model_ced.git)
    pip install transformers

## COALA

    cd external/
    git clone https://github.com/xavierfav/coala.git
    cd coala
    patch -p1 < ../../external/coala.patch
    cd ../..

## Dasheng

    pip install git+https://github.com/jimbozhang/hf_transformers_custom_model_dasheng.git

## ESResNe(X)t-fbsp

    cd external
    wget https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt
    git clone https://github.com/AndreyGuzhov/ESResNeXt-fbsp.git esresnext
    pip install msgpack_numpy
    cd esresnext
    sed -i 's/import ignite_trainer as it/#import ignite_trainer as it/' model/esresnet_base.py utils/transforms.py utils/datasets.py utils/datasets.py
    sed -i 's/it\.AbstractNet/torch.nn\.Module/' model/esresnet_base.py
    sed -i 's/it\.AbstractTransform/torch.nn\.Module/' utils/transforms.py
    sed -i 's/from model /from \. /' model/esresnet_base.py
    sed -i 's/from model\./from \./' model/esresnet_fbsp.py
    sed -i 's/from utils/from \.\.utils/' model/esresnet_base.py model/esresnet_fbsp.py
    sed -i 's/from utils/from \./' utils/datasets.py
    cd ../..

## HTS-AT

In addition to the following steps, please download the checkpoint as `external/HTSAT_AudioSet_Saved_1.ckpt` from https://github.com/RetroCirce/HTS-Audio-Transformer?tab=readme-ov-file#model-checkpoints.

    (cd external && git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git htsat)
    pip install h5py museval torchlibrosa

## M2D

To get M2D ready, follow the steps 👉 [M2D setup](https://github.com/nttcslab/m2d?tab=readme-ov-file#1-setup):

    cd external
    << follow the steps described in https://github.com/nttcslab/m2d?tab=readme-ov-file#1-setup >>

Download the weights from the GitHub. Example:

    wget https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip
    unzip m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip

You will find the `m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly` folder.
The following runs a linear evaluation on CREMA-D.

    python lineareval.py config/m2d.yaml cremad weight_file=m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth

## MS-CLAP, LAION-CLAP

    pip install msclap
    pip install laion-clap

## Opera

    (cd external && git clone https://github.com/evelyn0414/OPERA.git)
    (cd external/OPERA && curl -L -O https://huggingface.co/evelyn0414/OPERA/resolve/main/encoder-operaCT.ckpt)
    (cd external/OPERA && patch -p0 < ../opera.patch)

## VGGish

    cd external
    git clone https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch.git tcvrick_vggish
    sed -i 's/from audioset import/from \. import/' tcvrick_vggish/audioset/vggish_input.py
    wget https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases/download/v0.1/pytorch_vggish.zip
    unzip pytorch_vggish.zip
    cd ..

## WavCaps

In addition to the following steps, please download the checkpoint `HTSAT-BERT-PT.pt` in the folder `external/WavCaps` from https://github.com/XinhaoMei/WavCaps/tree/master/retrieval.

    (cd external && git clone https://github.com/XinhaoMei/WavCaps.git)
    (cd external/WavCaps && git apply ../../external/wavcaps.patch)
    pip install ruamel.yaml sentence_transformers wandb loguru torchlibrosa

