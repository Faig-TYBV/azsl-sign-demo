from pathlib import Path

folders = [
    "data/raw",
    "data/processed",
    "data/features",
    "models",
    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/utils",
    "outputs",
    "outputs/figures",
    "outputs/reports",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("Project structure created successfully!")

for folder in folders:
    print(f"✓ {folder}/")