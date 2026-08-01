def main() -> None:
    """Launch the loopback-only Streamlit application."""
    import sys
    from pathlib import Path

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
    ]
    streamlit_cli.main()
