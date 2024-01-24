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

from pathlib import Path
from multiprocessing import Pool
import fire
from tqdm import tqdm
import soundfile as sf
import librosa


def _converter_worker(args):
    subpathname, from_dir, to_dir, sample_rate, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/subpathname
    if verbose:
        print(from_dir, '->', to_name)

    # load wav
    wav, org_sr = sf.read(from_dir/subpathname, dtype='float32', always_2d=True)
    wav = wav.T  # (wave length, 1 or 2) -> (1 or 2, wave length)

    # stereo to mono (compatible with librosa)
    # ref: https://librosa.org/doc/main/generated/librosa.to_mono.html#librosa.to_mono
    wav = wav.mean(axis=0)

    # resample
    wav = librosa.resample(wav, orig_sr=org_sr, target_sr=sample_rate)

    # save wav
    to_name.parent.mkdir(exist_ok=True, parents=True)
    sf.write(to_name, data=wav, samplerate=sample_rate)  # subtype=sf.default_subtype('WAV') -- not always wav

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
