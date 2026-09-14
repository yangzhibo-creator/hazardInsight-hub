"""Seed the packaged knowledge index into the persistent demo directory once."""
import os
from pathlib import Path
import shutil

source = Path('/opt/demo-knowledge-bases')
target = Path('/app/backend/python/cluster-engine/artifacts/knowledge_bases')
target.mkdir(parents=True, exist_ok=True)
for index in source.iterdir():
    destination = target / index.name
    if index.is_dir() and not destination.exists():
        temporary = target / (index.name + '.initializing')
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(index, temporary)
        temporary.rename(destination)
os.execvp('python', ['python', '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'])
