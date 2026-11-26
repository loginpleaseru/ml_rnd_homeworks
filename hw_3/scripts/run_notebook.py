"""Execute hw3_analysis.ipynb in the selected installed Jupyter kernel."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="ml_rnd")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/hw3_analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("Выполнен:", path)


if __name__ == "__main__":
    main()
