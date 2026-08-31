"""Entry point: start the gateway."""

import uvicorn

from gateway.api import app  # noqa: F401
from gateway.config import get_config

if __name__ == "__main__":
    cfg = get_config().gateway
    uvicorn.run(app, host=cfg.host, port=cfg.port)
