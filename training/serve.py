"""Private local inference service for a trained Expert 802 checkpoint."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import torch
from generate import load_model
from tokenizer import load_tokenizer

def build_handler(model,tokenizer,device,auth_token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format_string,*args):
            print("Expert802 inference:",format_string%args)
        def do_GET(self):
            if self.path!="/health":
                return self.send_json(404,{"error":"Not found"})
            self.send_json(200,{"status":"ready","engine":"expert802-self-trained-transformer",
                "device":device,"parameters":model.parameter_count(),"context_length":model.config.block_size,
                "vocabulary_size":model.config.vocab_size})
        def do_POST(self):
            if self.path!="/generate":
                return self.send_json(404,{"error":"Not found"})
            if auth_token and self.headers.get("Authorization")!="Bearer "+auth_token:
                return self.send_json(401,{"error":"Unauthorized"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if size<1 or size>16000:
                    return self.send_json(400,{"error":"Request must be under 16 KB."})
                payload=json.loads(self.rfile.read(size).decode("utf-8"))
                prompt=payload.get("prompt","")
                if not isinstance(prompt,str) or not prompt.strip():
                    return self.send_json(400,{"error":"A prompt is required."})
                prompt_ids=tokenizer.encode(prompt).ids[-model.config.block_size:]
                configured_max=max(1,min(int(os.environ.get("EXPERT802_MAX_NEW_TOKENS","2048")),8192))\n                max_tokens=max(1,min(int(payload.get("max_new_tokens",1024)),configured_max))
                temperature=max(0.1,min(float(payload.get("temperature",0.8)),1.5))
                tokens=torch.tensor([prompt_ids],dtype=torch.long,device=device)
                generated=model.generate(tokens,max_new_tokens=max_tokens,temperature=temperature,stop_token_id=tokenizer.token_to_id("<|endoftext|>"))
                new_ids=generated[0].tolist()[len(prompt_ids):]
                answer=tokenizer.decode(new_ids,skip_special_tokens=True)
                self.send_json(200,{"reply":answer,"engine":"expert802-self-trained-transformer","generated_tokens":len(new_ids)})
            except (ValueError,TypeError,json.JSONDecodeError) as error:
                self.send_json(400,{"error":str(error)})
            except Exception as error:
                print("Inference failure:",repr(error))
                self.send_json(500,{"error":"Generation failed."})
        def send_json(self,status,data):
            body=json.dumps(data,ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return Handler

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="training/checkpoints/best.pt")
    parser.add_argument("--tokenizer",default="training/checkpoints/tokenizer.json")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8787)
    args=parser.parse_args()
    token=os.environ.get("EXPERT802_INFERENCE_TOKEN","")
    if args.host not in {"127.0.0.1","localhost","::1"} and len(token)<32:
        raise SystemExit("Remote binding requires EXPERT802_INFERENCE_TOKEN with at least 32 characters and a TLS reverse proxy.")
    device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model,payload=load_model(args.checkpoint,device)
    tokenizer=load_tokenizer(args.tokenizer)
    server=ThreadingHTTPServer((args.host,args.port),build_handler(model,tokenizer,device,token))
    print("Expert 802 private inference listening on",args.host,args.port,"step",payload.get("step"))
    server.serve_forever()

if __name__=="__main__":
    main()
