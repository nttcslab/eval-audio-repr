"""Audio file converter.

This converts the original audio files found in the source folder recursively,
then store under the destination folder with the same relative path structure.

The conversion process includes the following steps:
    - Stereo to mono
    - Resample to the sampling rate

Usage:
    python convert_wav.py /path/to/fsd50k work/16k/fsd50k 16000
    python convert_wav.py /path/to/speech_commands_v0.02 work/16k/spcv2 16000
    python convert_wav.py /data/A/VoxCeleb1 work/16k/vc1 16000
"""

from evar.common import (sys, Path, torch, torchaudio, AT)
from multiprocessing import Pool
import fire
from tqdm import tqdm


def _converter_worker(args):
    subpathname, from_dir, to_dir, sample_rate, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/subpathname
    if verbose:
        print(from_dir, '->', to_name)

    # load wav
    wav, org_sr = torchaudio.load(from_dir/subpathname)

    # stereo to mono (compatible with librosa)
    # ref: https://librosa.org/doc/main/generated/librosa.to_mono.html#librosa.to_mono
    wav = wav.mean(0, keepdims=True)

    # resample
    wav = AT.Resample(org_sr, sample_rate)(wav)

    # to int16
    wav = (wav * 32767.0).to(torch.int16)

    # save wav
    to_name.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(to_name, wav, sample_rate)

    return to_name.name


def convert_wav(from_dir, to_dir, sample_rate, suffix='.wav', verbose=False) -> None:
    from_dir = str(from_dir)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob(f'**/*{suffix}')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    print(f'Processing {len(files)} {suffix} files at a sampling rate of {sample_rate} Hz...')
    assert len(files) > 0

    with Pool() as p:
        args = [[f, from_dir, to_dir, sample_rate, verbose] for f in files]
        shapes = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    print('finished.')


if __name__ == "__main__":
    fire.Fire(convert_wav)
