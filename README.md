# DLO 2D Path Planning Presentation Assets

This project generates a 3-page PowerPoint and 2D pseudo-simulation assets for deformable linear object (DLO) path planning.

## Usage

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate the presentation:

```powershell
python make_ppt.py
```

Generate simulation animations, plots, and CSV analysis files:

```powershell
python generate_simulation_assets.py
```

Generated files are written to `outputs/`.
