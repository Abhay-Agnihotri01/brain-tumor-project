import json
import os

scripts = ["Colab_Code_GAN_Only", "Colab_Code_VAE_Only", "Colab_Code_Hybrid"]

for base in scripts:
    py_file = f"{base}.py"
    nb_file = f"{base}.ipynb"
    
    if not os.path.exists(py_file) or not os.path.exists(nb_file):
        print(f"Skipping {base} as files were not found.")
        continue
        
    print(f"Syncing {py_file} -> {nb_file}...")
    with open(py_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['source'] = lines
            break # assume all code is in first code cell

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
print("All Notebooks updated.")
