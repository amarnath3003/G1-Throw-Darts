"""SSH-only entry point for the Video to Throw PPO training archive."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiment2'))
import runtime_bootstrap
from train import main

if __name__ == '__main__':
    if sys.platform == 'win32':
        raise RuntimeError('Project training must run on the SSH server')
    if '--run-root' not in sys.argv:
        sys.argv.extend(['--run-root', str(ROOT/'experiment4/runs')])
    main()
