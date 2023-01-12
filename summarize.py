"""Summarize results for a model.
"""

from evar.common import (np, pd, Path, RESULT_DIR)
import fire


def get_weight(weight_file):
    weight_file = Path(weight_file)
    weight = weight_file.parent.name + '/' + weight_file.stem
    return weight


def available_tasks(df):
    ALL_TASKS = ['esc50', 'us8k', 'spcv2', 'vc1', 'voxforge', 'cremad', 'gtzan', 'nsynth', 'surge']
    tasks = [t for t in ALL_TASKS if t in list(df.columns)]
    return tasks


def summarize(weight_file, post=True):
    df = pd.read_csv(f'{RESULT_DIR}/scores.csv')
    df = df[df.report.str.contains(weight_file)]
    df['weight'] = get_weight(weight_file)
    src_df = df.copy()

    # summarize
    df = pd.pivot_table(df, index=['weight'], columns=['task'], values=['score'], aggfunc=np.mean)
    df.columns = df.columns.get_level_values(1)
    df = df[available_tasks(df)]
    if len(df) == 0:
        print(f'No data for {weight_file}.')
        return
    df['average'] = df.mean(1)

    # report
    report = df.applymap(lambda x: f'{x*100:.2f}%' if str(x).isnumeric else x).to_markdown()
    print(report)

    # save source results to a csv.
    report_csv = RESULT_DIR + '/' + str(df.index[0]).replace('/', '_') + '.csv'
    src_df.report = src_df.report.str.replace('\n', ' ')
    src_df.to_csv(report_csv, index=None)


if __name__ == '__main__':
    fire.Fire(summarize)
