"""Self-hosted, allowlisted Docker test runner. Never expose this service without authentication and TLS."""
import argparse
import json
import os
import subprocess
import tempfile
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from backend.cognition.code_intelligence import safe_path,InspectionError,MAX_FILES,MAX_FILE_BYTES,MAX_TOTAL_BYTES

PYTHON_IMAGE=os.environ.get("EXPERT802_PYTHON_IMAGE","python:3.11-slim")
NODE_IMAGE=os.environ.get("EXPERT802_NODE_IMAGE","node:22-alpine")
RUN_SEMAPHORE=threading.BoundedSemaphore(2)

def validate_files(files):
    if not isinstance(files,list) or not files or len(files)>MAX_FILES:
        raise ValueError("Supply between 1 and "+str(MAX_FILES)+" source files.")
    normalized=[]; seen=set(); total=0
    for item in files:
        if not isinstance(item,dict):
            raise ValueError("Each file must contain path and content.")
        path=safe_path(item.get("path"))
        if path in seen:
            raise ValueError("Duplicate path: "+path)
        seen.add(path)
        content=item.get("content")
        if not isinstance(content,str):
            raise ValueError("File content must be text.")
        size=len(content.encode("utf-8"))
        if size>MAX_FILE_BYTES:
            raise ValueError(path+" exceeds the per-file limit.")
        total+=size
        if total>MAX_TOTAL_BYTES:
            raise ValueError("Project snapshot exceeds 1 MB.")
        normalized.append((path,content))
    return normalized

def execute(files,command,target=""):
    normalized=validate_files(files)
    if command not in {"python_unittest","node_check"}:
        raise ValueError("Command is not in the sandbox allowlist.")
    paths={path for path,_ in normalized}
    if command=="python_unittest" and not any(path.startswith("tests/") for path in paths):
        raise ValueError("The snapshot must include a tests/ directory.")
    if command=="node_check":
        safe_target=safe_path(target)
        if safe_target not in paths or Path(safe_target).suffix.lower() not in {".js",".mjs",".cjs"}:
            raise ValueError("Choose an included .js, .mjs or .cjs file.")
    with tempfile.TemporaryDirectory(prefix="expert802-sandbox-") as temporary:
        root=Path(temporary)
        root.chmod(0o755)
        for relative,content in normalized:
            destination=root/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(content,encoding="utf-8")
            destination.chmod(0o644)
        container_name="expert802-"+uuid.uuid4().hex[:12]
        docker=["docker","run","--name",container_name,"--rm","--init","--network","none","--read-only",
                "--memory=768m","--cpus=1","--pids-limit=64",
                "--cap-drop=ALL","--security-opt=no-new-privileges",
                "--user=65534:65534","--workdir=/workspace",
                "--tmpfs","/tmp:rw,noexec,nosuid,size=64m",
                "--mount","type=bind,source="+str(root)+",target=/workspace,readonly"]
        if command=="python_unittest":
            docker += [PYTHON_IMAGE,"python","-B","-m","unittest","discover","-s","tests","-v"]
        else:
            docker += [NODE_IMAGE,"node","--check",target]
        started=time.monotonic()
        try:
            result=subprocess.run(docker,capture_output=True,text=True,timeout=45,check=False)
            return {"ok":result.returncode==0,"exit_code":result.returncode,"timed_out":False,
                    "duration_seconds":round(time.monotonic()-started,2),
                    "stdout":result.stdout[-12000:],"stderr":result.stderr[-12000:],
                    "command":command,"network":"disabled","source_mount":"read-only"}
        except subprocess.TimeoutExpired as error:
            try:
                subprocess.run(["docker","stop","--time","1",container_name],capture_output=True,text=True,timeout=5,check=False)
                subprocess.run(["docker","rm","-f",container_name],capture_output=True,text=True,timeout=5,check=False)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            return {"ok":False,"exit_code":None,"timed_out":True,
                    "duration_seconds":round(time.monotonic()-started,2),
                    "stdout":str(error.stdout or "")[-12000:],"stderr":"Execution exceeded 45 seconds.",
                    "command":command,"network":"disabled","source_mount":"read-only"}
        except FileNotFoundError:
            return {"ok":False,"exit_code":None,"timed_out":False,
                    "error":"Docker is not installed or the daemon is unavailable. Install Docker and preload the runner images."}

def build_handler(token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format_string,*args):
            print("Expert802 sandbox:",format_string%args)
        def do_GET(self):
            if self.path!="/health":
                return self.send_json(404,{"error":"Not found"})
            self.send_json(200,{"status":"ready","engine":"expert802-isolated-docker-runner",
                                "commands":["python_unittest","node_check"],"network_disabled":True})
        def do_POST(self):
            if self.path!="/run":
                return self.send_json(404,{"error":"Not found"})
            if not token or self.headers.get("Authorization")!="Bearer "+token:
                return self.send_json(401,{"error":"Unauthorized"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if size<1 or size>1100000:
                    return self.send_json(400,{"error":"Request must be under 1.1 MB."})
                payload=json.loads(self.rfile.read(size).decode("utf-8"))
                if not RUN_SEMAPHORE.acquire(blocking=False):
                    return self.send_json(429,{"error":"The isolated runner is busy. Try again shortly."})
                try:
                    result=execute(payload.get("files"),payload.get("command",""),payload.get("target",""))
                finally:
                    RUN_SEMAPHORE.release()
                status=200 if "error" not in result else 503
                self.send_json(status,result)
            except (ValueError,TypeError,InspectionError,json.JSONDecodeError,UnicodeDecodeError) as error:
                self.send_json(400,{"error":str(error)})
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
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8790)
    args=parser.parse_args()
    token=os.environ.get("EXPERT802_SANDBOX_TOKEN","")
    if len(token)<32:
        raise SystemExit("Set EXPERT802_SANDBOX_TOKEN to a random secret of at least 32 characters.")
    if args.host not in {"127.0.0.1","localhost","::1"}:
        print("Remote binding selected: terminate TLS at a trusted reverse proxy.")
    server=ThreadingHTTPServer((args.host,args.port),build_handler(token))
    print("Expert 802 isolated runner listening on",args.host,args.port)
    server.serve_forever()

if __name__=="__main__":
    main()
