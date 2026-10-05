"""Container entrypoint compatible with SageMaker's appended `serve` command."""

from __future__ import annotations

import os
import sys


def main() -> None:
    args = sys.argv[1:]
    if not args or args == ["serve"]:
        port = os.getenv("PORT", "8080")
        os.execvp(
            "uvicorn",
            [
                "uvicorn",
                "predict:app",
                "--host",
                "0.0.0.0",
                "--port",
                port,
                "--workers",
                "1",
                "--proxy-headers",
            ],
        )
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
