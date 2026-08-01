import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

from face_symmetry_lab import main


def test_main_launches_streamlit_on_loopback(monkeypatch) -> None:
    launched = False

    def fake_main() -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr(streamlit_cli, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["original"])

    main()

    assert launched is True
    assert sys.argv[0:2] == ["streamlit", "run"]
    assert Path(sys.argv[2]).name == "app.py"
    assert sys.argv[3:] == ["--server.address=127.0.0.1"]
