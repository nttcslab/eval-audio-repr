"""CirCor evaluation utility.

This program stratified-splits physionet.org/files/circor-heart-sound/1.0.3/training_data to heart-murmur-detection/data.
Then, it copies stratified data (under heart-murmur-detection/data) to evar working folder (under evar/work/16k).
It also creates metadata files as ../../evar/metadata/circor[1-3].csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import librosa
import torch
import torchaudio


## Copy raw data under physionet.org/files/circor-heart-sound/1.0.3/training_data to heart-murmur-detection/data
split_csvs = ['./datalist_stratified_data1.csv', './datalist_stratified_data2.csv', './datalist_stratified_data3.csv']
df = pd.concat([pd.read_csv(f) for f in split_csvs], ignore_index=True)

dest = Path('heart-murmur-detection/data')
for f in df.dest_file.values:
    f = Path(f)
    f.parent.mkdir(exist_ok=True, parents=True)
    from_file = Path('physionet.org/files/circor-heart-sound/1.0.3/training_data')/f.name
    #print('Copy', from_file, 'to', f)
    shutil.copy(from_file, f)


## Copy stratified data (under heart-murmur-detection/data) to evar working folder (evar/work/16k)
dfs = []

for split_no in [1, 2, 3]:
    trn = sorted(Path(f'heart-murmur-detection/data/stratified_data{split_no}/train_data/').glob('*.wav'))
    val = sorted(Path(f'heart-murmur-detection/data/stratified_data{split_no}/vali_data/').glob('*.wav'))
    tst = sorted(Path(f'heart-murmur-detection/data/stratified_data{split_no}/test_data/').glob('*.wav'))
    #Tr, V, Te = len(trn), len(val), len(tst)

    itrn = sorted(list(set([int(f.stem.split('_')[0]) for f in trn])))
    ival = sorted(list(set([int(f.stem.split('_')[0]) for f in val])))
    itst = sorted(list(set([int(f.stem.split('_')[0]) for f in tst])))
    Tr, V, Te = len(itrn), len(ival), len(itst)
    N = Tr + V + Te
    print(f'Split #{split_no} has samples: Training:{Tr}({Tr/N*100:.2f}%), Val:{V}({V/N*100:.2f}%), Test:{Te}({Te/N*100:.2f}%)')
    print(' Training sample IDs are:', itrn[:3], '...')

    df = pd.read_csv('physionet.org/files/circor-heart-sound/1.0.3/training_data.csv')

    def get_split(pid):
        if pid in itrn: return 'train'
        if pid in ival: return 'valid'
        if pid in itst: return 'test'
        assert False, f'Patient ID {pid} Unknown'
    df['split'] = df['Patient ID'].apply(get_split)


    SR = 16000
    L = int(SR * 5.0)
    STEP = int(SR * 2.5)

    ROOT = Path('physionet.org/files/circor-heart-sound/1.0.3/training_data/')
    TO_FOLDER = Path(f'../../work/16k/circor{split_no}')

    evardf = pd.DataFrame()

    for i, r in df.iterrows():
        pid, recloc, split, label = str(r['Patient ID']), r['Recording locations:'], r.split, r.Murmur
        # Not using recloc. Search real recordings...
        recloc = [f.stem.replace(pid+'_', '') for f in sorted(ROOT.glob(f'{pid}_*.wav'))]
        #print(pid, recloc, split, label)
        for rl in recloc:
            wav, sr = librosa.load(f'{ROOT}/{pid}_{rl}.wav', sr=SR)
            for widx, pos in enumerate(range(0, len(wav) - STEP + 1, STEP)):
                w = wav[pos:pos+L]
                org_len = len(w)
                if org_len < L:
                    w = np.pad(w, (0, L - org_len))
                    assert len(w) == L
                to_name = TO_FOLDER/split/f'{pid}_{rl}_{widx}.wav'
                to_rel_name = to_name.relative_to(TO_FOLDER)
                #print(pid, rl, len(wav)/SR, to_name, to_rel_name, org_len, len(w), pos)
                evardf.loc[to_name.stem, 'file_name'] = to_rel_name
                evardf.loc[to_name.stem, 'label'] = label
                evardf.loc[to_name.stem, 'split'] = split

                to_name.parent.mkdir(exist_ok=True, parents=True)
                w = torch.tensor(w * 32767.0).to(torch.int16).unsqueeze(0)
                torchaudio.save(to_name, w, SR)
    evardf.to_csv(f'../../evar/metadata/circor{split_no}.csv', index=None)
    print('Split', split_no)
    print(evardf[:3])

df[:3]
