"""Export only the unchanged Qwen3 decoder weights for the isolated vLLM worker."""
import argparse
import json
from pathlib import Path
import shutil
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    selected = {k:v for k,v in original.items() if k.startswith('model.layers.') or k in (
        'model.embed_tokens.weight','model.norm.weight','lm_head.weight')}
    if len(selected) != 399:
        raise ValueError(f'Expected 399 tensors for the local 36-layer decoder, got {len(selected)}')
    weight_map, total = {}, 0
    for index, filename in enumerate(sorted(set(selected.values()))):
        keys = [k for k,v in selected.items() if v == filename]
        with safe_open(source/filename, framework='pt', device='cpu') as f:
            tensors = {k:f.get_tensor(k) for k in keys}
        target = f'decoder-{index+1:05d}.safetensors'
        save_file(tensors, output/target, metadata={'format':'pt'})
        weight_map.update({k:target for k in tensors})
        total += sum(v.numel()*v.element_size() for v in tensors.values())
        print(target, len(tensors), flush=True)
        del tensors
    config = json.loads((source/'config.json').read_text())
    config['architectures'] = ['Qwen3ForCausalLM']
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'model.safetensors.index.json').write_text(json.dumps(
        {'metadata':{'total_size':total},'weight_map':weight_map},indent=2)+'\n')
    for filename in ['tokenizer_config.json','special_tokens_map.json','added_tokens.json',
                     'vocab.json','merges.txt','chat_template.jinja','generation_config.json']:
        if (source/filename).exists(): shutil.copy2(source/filename,output/filename)
    (output/'export.json').write_text(json.dumps({'source':str(source.resolve()),
        'tensor_count':len(selected),'total_bytes':total,'transformations':[]},indent=2)+'\n')

if __name__ == '__main__': main()
