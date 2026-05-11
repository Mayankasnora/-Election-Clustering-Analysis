import os, shutil

base = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base, 'plots'), exist_ok=True)
os.makedirs(os.path.join(base, 'data'), exist_ok=True)

for f in ['eci_results.csv', 'myneta.csv']:
    src = os.path.join(os.path.expanduser('~/Desktop'), f)
    dst = os.path.join(base, 'data', f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {f}")
    else:
        print(f"NOT FOUND on Desktop: {f}")

print("Setup complete.")
