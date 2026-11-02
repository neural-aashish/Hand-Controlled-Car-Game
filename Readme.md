# AI Hill Climb Controller

Control Hill Climb Racing with hand gestures using only a webcam.

| Gesture | Action | Key |
|---|---|---|
| Open palm | Accelerate | RIGHT ARROW |
| Closed fist | Brake / reverse | LEFT ARROW |
| No hand / unsure | Idle | none |

## Setup (Windows CMD, Python 3.11)
```cmd
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
The hand model downloads automatically on first run into `assets/`.

## Stop
Press Q or ESC in the preview window, or F12 in any window.

## Safety
Keys are released when the hand disappears, the webcam fails, an error
occurs, or the app closes. While the app runs, it types into whichever window
has focus, so focus the game before showing your hand.