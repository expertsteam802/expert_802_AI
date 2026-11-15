"""Simple health endpoint."""
import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "name": "Expert 802 AI", "language": "English", "engine": "project-owned-python"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
