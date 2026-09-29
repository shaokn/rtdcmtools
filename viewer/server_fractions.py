"""Launch the viewer with the fraction-organized NIfTI collection."""
import argparse
from pathlib import Path

import server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nifti-root', type=Path,
                        default=server.OUTPUT_ROOT / 'nifti_fractions')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    server.NIFTI_ROOT = args.nifti_root.resolve()
    server.app.config['DEFAULT_SOURCE'] = 'nifti'
    print(f'Fraction viewer: http://127.0.0.1:{args.port}/?source=nifti', flush=True)
    server.app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
