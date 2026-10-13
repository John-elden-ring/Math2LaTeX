"""Top-level CLI launcher for Math2LaTeX.

Enables running directly via:
    python app/cli.py photo.jpg -o notes.tex
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app.cli import main

if __name__ == "__main__":
    main()
