#!/usr/bin/env python3
"""Dev server for web/, stdlib only.

Serves this directory (index.html, worker.js, pkg/, data/, models/) with
permissive CORS. No COOP/COEP headers -- this demo's backend is CPU
(burn-ndarray) via wasm-bindgen, no SharedArrayBuffer / cross-origin
isolation requirement (that's a `wgpu`/threaded-wasm concern, not
applicable here yet -- see crates/t0-wasm/README.md).

Usage:
    python3 web/serve.py [--port PORT] [--bind BIND]
"""
import argparse
import http.server
import os
import socketserver
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class DemoPageHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    handler = lambda *a, **kw: DemoPageHandler(*a, directory=ROOT, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {ROOT} on http://{args.bind}:{args.port} (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
