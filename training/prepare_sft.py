"""Build a held-out-by-conversation token dataset using the base model tokenizer."""
import argparse
import json
import random
import shutil
from pathlib import Path
import numpy as np
from tokenizer import load_tokenizer

def prepare(input_path,output_dir,tokenizer_path,validation_fraction=0.05,seed=802):
    randomizer=random.Random(seed)
    training,validation=[],[]
    tokenizer=load_tokenizer(tokenizer_path)
    separator=tokenizer.token_to_id("<|endoftext|>")
    source=Path(input_path)
    for line in source.read_text(encoding="utf-8",errors="ignore").splitlines():
        try:
            record=json.loads(line)
        except json.JSONDecodeError:
            continue
        messages=record.get("messages",[]) if isinstance(record,dict) else []
        turns=[]
        for message in messages if isinstance(messages,list) else []:
            if not isinstance(message,dict) or message.get("role") not in {"system","user","assistant"}:
                continue
            body=message.get("content","")
            if isinstance(body,str) and body.strip():
                turns.append("<|"+message["role"]+"|> "+body.strip())
        if not turns:
            continue
        ids=tokenizer.encode("\n".join(turns)).ids+[separator]
        split=record.get("split") if isinstance(record,dict) else None
        if split=="validation" or (split not in {"train","validation"} and randomizer.random()<validation_fraction):
            validation.extend(ids)
        else:
            training.extend(ids)
    if len(training)<=513 or len(validation)<=513:
        raise ValueError("Supply enough complete conversations for both splits (more than 513 tokens each).")
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(tokenizer_path,output/"tokenizer.json")
    dtype=np.uint16 if tokenizer.get_vocab_size()<=65535 else np.uint32
    np.asarray(training,dtype=dtype).tofile(output/"train.bin")
    np.asarray(validation,dtype=dtype).tofile(output/"val.bin")
    metadata={"tokenizer":"byte-level-bpe-v1","vocab_size":tokenizer.get_vocab_size(),
              "token_dtype":str(np.dtype(dtype)),"train_tokens":len(training),"validation_tokens":len(validation)}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--tokenizer",default="training/data/tokenizer.json")
    parser.add_argument("--output",default="training/sft_data")
    args=parser.parse_args()
    print(json.dumps(prepare(args.dataset,args.output,args.tokenizer),indent=2))
