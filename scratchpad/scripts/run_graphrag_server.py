from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("PZ_GRAPH_UI_HOST", "127.0.0.1")
    port = int(os.getenv("PZ_GRAPH_UI_PORT", "8002"))

    uvicorn.run(
        "palimpzest.server.graphrag_app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
