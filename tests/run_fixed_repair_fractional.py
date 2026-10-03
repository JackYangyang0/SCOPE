"""Independent signed/fractional alpha-beta checks on repaired regression outputs."""
import argparse
import json
import subprocess
from pathlib import Path
from SCOPE.utils.common_utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--single', action='store_true', help='Check this one code directory instead of case1/2/3')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vcvars = Path('D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat')
    records = []
    cases = [args.output] if args.single else sorted(args.output.glob('case*'))
    for case in cases:
        exe = case/'fractional_check.exe'
        command = f'call "{vcvars}" && nvcc -O3 -arch=sm_89 -std=c++17 -I"{case}" "{root / "tests/cuda_check_alpha_beta.cu"}" -lcublas -o "{exe}" && "{exe}"'
        result = subprocess.run('cmd /d /c ' + command, capture_output=True, text=True,
                                errors='replace', timeout=180)
        records.append({'case':case.name, 'returncode':result.returncode,
                        'stdout':result.stdout, 'stderr':result.stderr})
        print(json.dumps({'case':case.name, 'returncode':result.returncode,
                          'output':result.stdout[-900:]}, ensure_ascii=True), flush=True)
        save_json(args.output/'fractional_checks.json', {'records':records})
    if len(records) != (1 if args.single else 3) or any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
