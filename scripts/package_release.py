"""Make an uploadable archive with source/docs only, excluding local environments and data."""
import hashlib
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "jev-subspace-evolve.zip"
    output.parent.mkdir(exist_ok=True)
    files = [root / name for name in ("README.md", "requirements.txt", "pyproject.toml", ".gitignore")]
    for directory in ("configs", "src", "scripts", "tests", "docs"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
                     and ".egg-info" not in str(p))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(files):
            relative = file.relative_to(root).as_posix()
            # Ship reference configs, not users' local paths or later edits containing credentials.
            if relative.startswith("configs/") and relative != "configs/autodl_4090.json":
                continue
            archive.write(file, "self-evolve/" + relative)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(checksum + "  " + output.name + "\n", encoding="utf-8")
    print(f"{output}\n{output.stat().st_size} bytes\nSHA256 {checksum}")


if __name__ == "__main__":
    main()
