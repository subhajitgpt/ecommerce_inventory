import os

from hitpromo import build_app


def main() -> None:
    app = build_app()
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8005")), debug=True)


if __name__ == "__main__":
    main()
