"""Verify and relocate a knowledge index when only the model's disk path changed."""
import json
from pathlib import Path
from retrain_cluster.config import Catalog, Settings, model_fingerprint, verify_local_model
from retrain_cluster.retrieval.chroma import ChromaRetriever

original = json.loads(Path('/opt/source-model.json').read_text(encoding='utf-8'))
current = Catalog(Settings.load('cluster-engine/configs/app.toml')).model(original['model_id'])
assert {k: v for k, v in original.items() if k != 'source'} == {
    k: v for k, v in current.items() if k != 'source'
}, 'Only the model directory may change during relocation'
verify_local_model(current)
old_fingerprint = model_fingerprint(original)
new_fingerprint = model_fingerprint(current)
root = Path('/opt/demo-knowledge-bases')
for directory in root.iterdir():
    if not directory.is_dir():
        continue
    # This validates count, metric and the checksum of every stored embedding.
    retriever = ChromaRetriever(root, directory.name, old_fingerprint, current['dimension'])
    manifest = retriever.manifest
    manifest['model_fingerprint'] = new_fingerprint
    manifest['relocation'] = {
        'reason': 'identical checked model files and encoding options; filesystem path changed',
        'original_model_fingerprint': old_fingerprint,
        'original_source': original['source'],
        'container_source': current['source'],
    }
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Verified and relocated {directory.name}: {manifest["count"]} records')
