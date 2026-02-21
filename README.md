<<<<<<< HEAD
# ✏️ AIR DRAWING SYSTEM

Draw on a virtual canvas using just your **index finger** and a webcam — no mouse, no touch screen needed!

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python air_drawing.py
```

---

## 🖐️ Hand Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ **Index finger only (up)** | Draw / Erase on canvas |
| ✌️ **Index + Middle (up)** | Selection mode — hover over toolbar to pick color |
| ✊ **Fist / other** | Idle — lifts pen without drawing |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `S` | Save current drawing to `saved_drawings/` folder |
| `C` | Clear the entire canvas |
| `Q` | Quit the application |

---

## 🎨 Color Palette

The toolbar at the top of the screen contains:

- 🔴 Red
- 🟠 Orange
- 🟡 Yellow
- 🟢 Green
- 🔵 Blue
- 🟣 Purple
- ⬜ White
- 🧹 **Eraser** (last slot — removes drawn pixels)

**To change color/tool:** Raise both index and middle fingers (Selection mode), then hover your index fingertip over the color you want in the toolbar.

---

## 💾 Saving Drawings

Press `S` at any time. The drawing is saved as a PNG in the `saved_drawings/` folder with a timestamped filename like:

```
saved_drawings/drawing_20240221_143022.png
```

---

## 📁 Project Structure

```
air_drawing_system/
├── air_drawing.py      # Main application
├── requirements.txt    # Python dependencies
└── README.md           # This file
saved_drawings/         # Auto-created when you save
```

---

## ⚙️ Requirements

- Python 3.8+
- Webcam (built-in or USB)
- Good lighting for best hand detection

---

## 🛠️ Customization Tips

In `air_drawing.py`, you can tweak these constants at the top:

| Variable | Default | Description |
|---|---|---|
| `BRUSH_THICKNESS` | `8` | Drawing line width |
| `ERASER_THICKNESS` | `40` | Eraser circle size |
| `CANVAS_ALPHA` | `0.6` | Drawing overlay transparency |
| `HEADER_HEIGHT` | `100` | Toolbar height in pixels |

---

## 🔧 Troubleshooting

**Camera not found:** Make sure no other app is using the camera. Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**Hand not detected:** Improve lighting, keep your hand clearly in frame, and make sure your background isn't too cluttered.

**Laggy performance:** Lower the camera resolution in `cap.set(...)` calls (e.g., 640×480).
=======
# air-drawing-system
>>>>>>> 5689599c161fc454a402430d2b20b081c43fac99
