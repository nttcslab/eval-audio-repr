# Example command lines for evaluating models

## AST

    python 2pass_blackbox.py config/ast.yaml fsd50k
    python 2pass_blackbox.py config/ast.yaml spcv2
    python 2pass_blackbox.py config/ast.yaml us8k
    python 2pass_blackbox.py config/ast.yaml surge --lr=0.0001
    python 2pass_blackbox.py config/ast.yaml nsynth
    python 2pass_blackbox.py config/ast.yaml nspitch
    python 2pass_blackbox.py config/ast.yaml vc1
    python 2pass_blackbox.py config/ast.yaml cremad
    python 2pass_blackbox.py config/ast.yaml voxforge
    python 2pass_blackbox.py config/ast.yaml esc50
    python 2pass_blackbox.py config/ast.yaml gtzan batch_size=12

## BYOL-A

    python 2pass_blackbox.py config/byola.yaml fsd50k
    python 2pass_blackbox.py config/byola.yaml spcv2
    python 2pass_blackbox.py config/byola.yaml us8k
    python 2pass_blackbox.py config/byola.yaml surge --lr=0.0001
    python 2pass_blackbox.py config/byola.yaml nsynth
    python 2pass_blackbox.py config/byola.yaml nspitch
    python 2pass_blackbox.py config/byola.yaml vc1
    python 2pass_blackbox.py config/byola.yaml cremad
    python 2pass_blackbox.py config/byola.yaml voxforge
    python 2pass_blackbox.py config/byola.yaml esc50
    python 2pass_blackbox.py config/byola.yaml gtzan batch_size=64 --lr=0.001

## PANNs' CNN14

    python 2pass_blackbox.py config/cnn14.yaml cremad
    python 2pass_blackbox.py config/cnn14.yaml voxforge
    python 2pass_blackbox.py config/cnn14.yaml esc50
    python 2pass_blackbox.py config/cnn14.yaml gtzan
    python 2pass_blackbox.py config/cnn14.yaml fsd50k
    python 2pass_blackbox.py config/cnn14.yaml nsynth --lr=0.00001
    python 2pass_blackbox.py config/cnn14.yaml nspitch
    python 2pass_blackbox.py config/cnn14.yaml surge
    python 2pass_blackbox.py config/cnn14.yaml vc1
    python 2pass_blackbox.py config/cnn14.yaml spcv2
    python 2pass_blackbox.py config/cnn14.yaml us8k

## VGGish

    python 2pass_blackbox.py config/vggish.yaml fsd50k
    python 2pass_blackbox.py config/vggish.yaml nsynth
    python 2pass_blackbox.py config/vggish.yaml nspitch
    python 2pass_blackbox.py config/vggish.yaml surge
    python 2pass_blackbox.py config/vggish.yaml vc1 --lr=0.0005
    python 2pass_blackbox.py config/vggish.yaml spcv2
    python 2pass_blackbox.py config/vggish.yaml us8k
    python 2pass_blackbox.py config/vggish.yaml cremad
    python 2pass_blackbox.py config/vggish.yaml voxforge
    python 2pass_blackbox.py config/vggish.yaml esc50 --lr=0.003
    python 2pass_blackbox.py config/vggish.yaml gtzan batch_size=128

