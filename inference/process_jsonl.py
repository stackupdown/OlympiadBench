import os
import json
path = '~/labs/202504-longcot/OlympiadBench/inference/generated/OE_TO_maths_en_COMP/s1_epoch3_distill/Qwen2.5-3B-Open-R1-Distill.jsonl'
path = os.path.expanduser(path)
def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl(path)
print(data[0].keys())

def dump_json(data, path):
    with open(path, 'w') as f:
        f.write(json.dumps(data) + '\n')
    print("dump to {}".format(path))

dirname = os.path.dirname(path)
basename = os.path.basename(path)
output_path = os.path.join(dirname, basename + '.json')
dump_json(data, output_path)
