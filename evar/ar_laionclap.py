"""Wrapper code for:

Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation

## Reference
- [1] https://arxiv.org/abs/2211.06687
- [2] https://github.com/LAION-AI/CLAP
"""

from evar.ar_base import BaseCLAP
from packaging import version
import torch
import transformers, os
try:
    import laion_clap
except:
    pass  # please install: pip install laion-clap


def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    # https://github.com/LAION-AI/CLAP/blob/817041c079af560fa2c610287c68c7c97ace50b6/src/laion_clap/clap_module/factory.py#L53
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # removing position_ids to maintain compatibility with latest transformers update
        if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]
    return state_dict


def load_ckpt(self, ckpt = None, model_id = -1, verbose = True):
    # https://github.com/LAION-AI/CLAP/blob/817041c079af560fa2c610287c68c7c97ace50b6/src/laion_clap/hook.py#L74C2-L119C5
    """Load the pretrained checkpoint of CLAP model

    Parameters
    ----------
    ckpt: str
        if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n 
        For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
    model_id:
        if model_id is specified, you can download our best ckpt, as:
            id = 0 --> 630k non-fusion ckpt \n
            id = 1 --> 630k+audioset non-fusion ckpt \n
            id = 2 --> 630k fusion ckpt \n
            id = 3 --> 630k+audioset fusion ckpt \n
        Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
    """
    import wget
    download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
    download_names = [
        '630k-best.pt',
        '630k-audioset-best.pt',
        '630k-fusion-best.pt',
        '630k-audioset-fusion-best.pt'
    ]
    if ckpt is not None:
        print(f'Load the specified checkpoint {ckpt} from users.')
    else:
        print(f'Load our best checkpoint in the paper.')
        if model_id == -1:
            model_id = 3 if self.enable_fusion else 1
        package_dir = os.path.dirname(os.path.realpath(__file__))
        weight_file_name = download_names[model_id]
        ckpt = os.path.join(package_dir, weight_file_name)
        print(ckpt)
        if os.path.exists(ckpt):
            print(f'The checkpoint is already downloaded')
        else:
            print('Downloading laion_clap weight files...')
            ckpt = wget.download(download_link + weight_file_name, os.path.dirname(ckpt))
            print('Download completed!')
    print('Load Checkpoint...')
    ckpt = load_state_dict(ckpt, skip_params=True)
    self.model.load_state_dict(ckpt)
    if verbose:
        param_names = [n for n, p in self.model.named_parameters()]
        for n in param_names:
            print(n, "\t", "Loaded" if n in ckpt else "Unloaded")


class AR_LAIONCLAP(BaseCLAP):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = laion_clap.CLAP_Module()
        # workaround to make sure: del state_dict["text_branch.embeddings.position_ids"]
        print(version.parse(transformers.__version__))
        self.backbone.load_ckpt = load_ckpt.__get__(self.backbone, laion_clap.CLAP_Module)  
        self.backbone.load_ckpt()

    def encode_frames(self, batch_audio):
        assert False, 'encode_frames for LAION-CLAP is not supported for now'

    def forward(self, batch_audio):
        audio_embeddings = self.backbone.get_audio_embedding_from_data(x=batch_audio, use_tensor=True)
        return audio_embeddings

    def encode_audio(self, batch_audio):
        audio_embeddings = self.forward(batch_audio)
        return audio_embeddings

    def encode_text(self, batch_text):
        text_embeddings = self.backbone.get_text_embedding(batch_text, use_tensor=True)
        return text_embeddings
