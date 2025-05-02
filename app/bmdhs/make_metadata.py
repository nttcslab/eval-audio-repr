# Generate EVAR metadata csv files.
import pandas as pd


splits = [pd.read_csv(f) for f in ['split1.csv', 'split2.csv', 'split3.csv']]

for df_index, spl_df in enumerate(splits):
    d = pd.DataFrame()
    for index, row in spl_df.iterrows():
        labels, split = row['label'], row['split']
        for filestem in row[[f'recording_{i}' for i in range(1, 8+1)]].values:
            d = pd.concat([d, pd.DataFrame({'file_name': [f'train/{filestem}.wav'], 'label': [labels], 'split': [split]})])
    d.to_csv(f'../../evar/metadata/bmdhs{df_index + 1}.csv')
    print(d[:3])
