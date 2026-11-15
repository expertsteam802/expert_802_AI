"""Optional authenticated client for the private Docker test runner."""
import json
import os
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse,urlunparse
from urllib.request import Request,urlopen

class SandboxRuntime:
    def __init__(self):
        self.url=os.environ.get("EXPERT802_SANDBOX_URL","").strip()
        self.token=os.environ.get("EXPERT802_SANDBOX_TOKEN","").strip()
        parsed=urlparse(self.url)
        local=parsed.hostname in {"127.0.0.1","localhost","::1"}\n        self.configured=bool(parsed.netloc and (parsed.scheme=="https" or (parsed.scheme=="http" and local)) and len(self.token)>=32)

    def health(self):
        if not self.configured:
            return {"configured":False,"ready":False}
        parsed=urlparse(self.url)
        path=parsed.path.rsplit("/",1)[0]+"/health"
        health_url=urlunparse((parsed.scheme,parsed.netloc,path,"","",""))
        try:
            with urlopen(Request(health_url,headers={"Authorization":"Bearer "+self.token}),timeout=2) as response:
                data=json.loads(response.read().decode("utf-8"))
            return {"configured":True,"ready":response.status==200 and data.get("status")=="ready",
                    "commands":data.get("commands",[])}
        except (HTTPError,URLError,TimeoutError,ValueError,OSError):
            return {"configured":True,"ready":False}

    def run(self,files,command,target=""):
        if not self.health()["ready"]:
            return {"ok":False,"error":"The private isolated sandbox is not ready."}
        body=json.dumps({"files":files,"command":command,"target":target}).encode("utf-8")
        request=Request(self.url,data=body,headers={"Content-Type":"application/json",
            "Authorization":"Bearer "+self.token},method="POST")
        try:
            with urlopen(request,timeout=50) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError,URLError,TimeoutError,ValueError,OSError) as error:
            return {"ok":False,"error":"Isolated execution failed: "+str(error)[:240]}
