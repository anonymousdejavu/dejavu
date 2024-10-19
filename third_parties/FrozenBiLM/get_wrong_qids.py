import argparse
import pandas as pd
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv-path", type=str, required=True)
    parser.add_argument('--input-json-path', type=str, required=True)
    parser.add_argument('--output-csv-path', type=str, required=True)
    args = parser.parse_args()

    wrong_qids = []
    with open(args.input_json_path, 'r') as f:
        data = json.load(f)
        for q_id, items in data.items():
            if not items['acc']:
                wrong_qids.append(int(q_id))

    df = pd.read_csv(args.input_csv_path)
    original_len = len(df)
    df = df[df['qid'].isin(wrong_qids)]
    print(f"Original length: {original_len}, New length: {len(df)}")
    df.to_csv(args.output_csv_path, index=False)