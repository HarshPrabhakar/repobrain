from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run RepoBrain's local FastAPI server."
            )
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default keeps RepoBrain local-only."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port.",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Enable development reload."
        ),
    )

    return (
        parser.parse_args()
    )


def main() -> int:

    args = (
        parse_args()
    )

    uvicorn.run(
        "repobrain.api.app:app",
        host=(
            args.host
        ),
        port=(
            args.port
        ),
        reload=(
            args.reload
        ),
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
