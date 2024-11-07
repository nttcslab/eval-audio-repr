"""Summarize results for a model.
"""

from evar.common import (np, pd, Path, RESULT_DIR)
import fire


def get_weight(weight_file):
    weight_file = Path(weight_file)
    weight = weight_file.parent.name + '/' + weight_file.stem
    return weight


def available_tasks(df):
    ALL_TASKS = ['esc50', 'us8k', 'spcv2', 'vc1', 'voxforge', 'cremad', 'gtzan', 'nsynth', 'surge', 'fsd50k'] \
        + ['zs_esc50', 'zs_us8k', 'zs_spcv2', 'zs_vc1', 'zs_voxforge', 'zs_cremad', 'zs_gtzan', 'zs_nsynth', 'zs_surge', 'zs_fsd50k', 'zs_as']
    tasks = [t for t in ALL_TASKS if t in list(df.columns)]
    return tasks


def summarize(weight_file, post=True):
    # summarize LE
    df = pd.read_csv(f'{RESULT_DIR}/scores.csv')
    df = df[df.report.str.contains(weight_file)]
    df['weight'] = get_weight(weight_file)
    src_df = df.copy()

    df = pd.pivot_table(df, index=['weight'], columns=['task'], values=['score'], aggfunc=np.mean)
    df.columns = df.columns.get_level_values(1)
    df = df[available_tasks(df)]
    if len(df) == 0:
        print(f'No data for {weight_file}.')
        return
    df['average'] = df.mean(1)

    # summarize ATR
    if Path(f'{RESULT_DIR}/scores.csv').exists():
        d = pd.read_csv(f'{RESULT_DIR}/retrieval_scores.csv')
        d = d[d.weight.str.contains(weight_file, na=False)]
        if len(d) > 0:
            d = d.set_index('model')
            d['weight'] = get_weight(weight_file)
            d.columns = ['task', 'a2tR1', 'a2tR5', 'a2tR10', 't2aR1', 't2aR5', 't2aR10', 'weight']
            new_d = None
            for t, shortname in [('audiocaps', 'A'), ('clotho', 'C')]:
                d_ = d[d.task == t][['a2tR1', 'a2tR5', 'a2tR10', 't2aR1', 't2aR5', 't2aR10']]
                d_.columns = [shortname + c for c in list(d_.columns)]
                d_.index = ['same_index']
                new_d = d_ if new_d is None else pd.concat([new_d, d_], axis=1)
            new_d['weight'] = get_weight(weight_file)
            new_d = new_d.set_index('weight') * 0.01
            df = pd.concat([df, new_d], axis=1)

    # report
    report = df.applymap(lambda x: f'{x*100:.2f}%' if str(x).isnumeric else x).to_markdown()
    print(report)

    # save source results to a csv.
    report_csv = RESULT_DIR + '/' + str(df.index[0]).replace('/', '_') + '.csv'
    src_df.report = src_df.report.str.replace('\n', ' ')
    src_df.to_csv(report_csv, index=None)


if __name__ == '__main__':
    fire.Fire(summarize)
