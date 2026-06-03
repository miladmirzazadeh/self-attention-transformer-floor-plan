# self-attention-transformer-floor-plan

A **self-attention transformer that labels CAD floor-plan primitives** (PrimitiveNet),
trained on synthetic plans from the Vitruev generator. Each primitive (line / arc /
circle / polyline segment) is classified as **wall / door / window / column / clutter**
using its geometry + global self-attention over the plan. No images — works directly
on vector geometry, so coordinates stay exact.

```
generator (scenarios) ──▶ DXF (temp) ──▶ labeled primitives (JSON) ──▶ PrimitiveNet
        layout.py            ezdxf            parse_dxf.py                model.py
```

## Layout
- `generator/` — the Vitruev synthetic generator (scenario → DXF).
- `primitivenet/` — the pipeline:
  - `make_dataset.py` — **streaming** generate+parse (scenario batches → labeled primitives, no DXF pile).
  - `parse_dxf.py` — DXF → per-primitive features + labels (by layer: `A-WALL-*`→wall, `A-DOOR`→door, `A-GLAZ`→window, `A-COLS`→column, else clutter).
  - `model.py` — `PrimitiveNet` (4-layer set-transformer, 17-dim features, 5 classes).
  - `dataset.py`, `train.py` — node-classification training (cosine LR, clutter down-weight, door/window up-weight, time budget, resume).

## 1. Prepare the dataset (local — you have the generator + configs)
```bash
pip install -r requirements.txt
PYTHONPATH=. python -m primitivenet.make_dataset \
    --configs /path/to/Vitruev_synthdata/configs --out prim_ds
tar czf primitives.tar.gz -C prim_ds train val      # ~240 MB for 10k plans
```
Upload `primitives.tar.gz` to a Kaggle dataset (e.g. `floorplan-primitives`).

## 2. Train (Kaggle free GPU or RunPod)
Attach the dataset, then:
```bash
python -m primitivenet.train --data prim_ds --num-classes 5 \
    --epochs 80 --batch 16 --open-weight 2 --time-budget 8 --out runs/primnet
```
`best.pt` is ~12 MB. Watch `mIoU [clutter wall door window column]`.

## 3. Inference (real DXF → labeled geometry)
Run `parse_dxf.primitives_of()` on a DXF → `PrimitiveNet` → per-primitive labels →
keep wall/door/window with their **exact original coordinates** → write a clean DXF.
