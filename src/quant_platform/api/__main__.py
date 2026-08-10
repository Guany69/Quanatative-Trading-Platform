"""Development entry point bound to loopback by default."""

import uvicorn

from quant_platform.api.config import ApiSettings


def main() -> None:
    settings = ApiSettings()
    uvicorn.run(
        "quant_platform.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
