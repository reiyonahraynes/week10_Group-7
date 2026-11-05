import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Sandbox:

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="week10_"))
        shutil.copytree(ROOT / "data", self.dir / "data")
        shutil.copy(ROOT / "config.json", self.dir / "config.json")
        self.config_path = self.dir / "config.json"
        self.outputs = self.dir / "outputs"
        self.staging = self.dir / "staging"

    def edit_config(self, **changes) -> None:
        cfg = json.loads(self.config_path.read_text())
        cfg.update(changes)
        self.config_path.write_text(json.dumps(cfg, indent=2))

    def edit_manifest(self, fn) -> None:
        path = self.dir / "data" / "manifest.json"
        manifest = json.loads(path.read_text())
        fn(manifest)
        path.write_text(json.dumps(manifest, indent=2))

    def spec(self, filename: str):
        manifest = json.loads((self.dir / "data" / "manifest.json").read_text())
        return next(s for s in manifest["files"] if s["file"] == filename)

    def drop_integrity(self, filename: str) -> None:
        def fn(m):
            for s in m["files"]:
                if s["file"] == filename:
                    s.pop("sha256", None)
                    s.pop("bytes", None)
        self.edit_manifest(fn)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)