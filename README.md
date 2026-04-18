# AI Fitness Trainer

A Python-based posture detection system for guided workouts. The app uses Streamlit for the frontend, `streamlit-webrtc` for webcam streaming, and MediaPipe Pose landmarks to deliver exercise-specific feedback for:

- Squats
- Push-ups
- Planks

## Features

- Real-time pose overlay on top of the camera feed
- Photo upload posture analysis
- Video upload posture analysis with downloadable processed output
- 20 gym exercises and 20 yoga asana exercises in the selector
- Every exercise now includes a built-in image card, video tutorial link, benefits, and step-by-step guidance
- Rep counting for squats and push-ups
- Hold timer for planks
- Live form score and coaching cues
- Exercise-specific training tips based on detected form
- Python frontend with a single app entrypoint

## Run It

Install the dependencies:

```powershell
cd "d:\Fitness Traniner"
pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

## Notes

- The main Python frontend lives in `app.py`.
- The older static prototype files (`index.html`, `styles.css`, `app.js`) are still in the repo, but the Python app is now the preferred way to run the project.
- You can use the app with live webcam input or upload a photo/video for offline posture review.
- For side-profile exercises like push-ups and planks, place the camera to your side.
- This is a prototype heuristic coach, not a medical or biomechanical diagnostic tool.
