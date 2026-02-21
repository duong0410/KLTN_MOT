# Models Directory

Place your trained YOLO model files here.

## Expected Files

- `yolo11s_traffic.pt` - Your trained YOLO11s model

## Download Models

If you don't have a trained model, you can:

1. **Use pre-trained YOLO models** from Ultralytics:
   ```bash
   # Will auto-download on first use
   python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"
   ```

2. **Train your own model** on traffic/vehicle dataset

3. **Copy your existing model** to this directory

## File Structure

```
models/
├── yolo11s_traffic.pt    # Your trained model
├── yolo11n.pt            # (optional) Nano model
└── yolo11m.pt            # (optional) Medium model
```

## Note

Large model files (*.pt) are excluded from git by default (see .gitignore).
