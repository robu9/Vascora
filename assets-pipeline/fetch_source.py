"""Download the exact verified source archive; reject upstream drift."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'provenance.json').read_text())
directory = root / 'source'
directory.mkdir(exist_ok=True)
archive = directory / 'Z-Anatomy.zip'
if not archive.exists():
    urllib.request.urlretrieve(manifest['source_url'], archive)
actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != manifest['source_sha256']:
    raise SystemExit('Source checksum mismatch. Obtain the pinned archive; do not silently rebuild from changed anatomy.')
with zipfile.ZipFile(archive) as source:
    source.extract('Z-Anatomy/Startup.blend', directory)
print('Verified source:', actual)
