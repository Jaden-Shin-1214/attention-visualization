"""Dev server with caching disabled, so every reload fetches the latest files.
Usage: python3 serve.py [port]   (default 8021, binds 127.0.0.1)
"""
import http.server, socketserver, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8021


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class Server(socketserver.TCPServer):
    allow_reuse_address = True


with Server(("127.0.0.1", PORT), NoCacheHandler) as httpd:
    print(f"serving (no-cache) at http://127.0.0.1:{PORT}")
    httpd.serve_forever()
