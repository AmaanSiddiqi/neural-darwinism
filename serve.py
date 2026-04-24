"""
Launch the Colony web server.
Usage: python serve.py [--port 8000]
Then open http://localhost:8000
"""
import argparse
import os

# Must be first — sets HSA env var before torch loads
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Colony UI → http://localhost:{args.port}")
    uvicorn.run("colony.api.server:app", host=args.host, port=args.port, reload=False)
