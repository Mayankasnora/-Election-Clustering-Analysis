"""Run the full WB Electoral Clustering Analysis"""
import subprocess, sys, os
base = os.path.dirname(os.path.abspath(__file__))

for part in ['analysis_p1.py', 'analysis_p2.py']:
    print(f"\n{'='*50}\nRunning {part}\n{'='*50}")
    result = subprocess.run([sys.executable, os.path.join(base, part)], cwd=base)
    if result.returncode != 0:
        print(f"ERROR in {part}")
        sys.exit(1)

print("\n✅ Full analysis complete! Check the 'plots/' folder for all 12 charts.")
