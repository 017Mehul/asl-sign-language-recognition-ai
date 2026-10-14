# Frontend-only ASL recognition

This static frontend runs sign recognition directly in the browser. It does not call the FastAPI backend for camera frames, predictions, sentence editing, or speech.

## Model files

- Keep the original Keras model at `../models/asl_model.h5`.
- Keep the output label order at `../models/labels.json`.
- GitHub Actions converts the H5 model to TensorFlow.js format and commits generated files into `frontend/models/asl_tfjs/`.
- The browser loads `./models/asl_tfjs/model.json`; the generated binary weight shards must sit beside it.

The conversion workflow runs when `models/asl_model.h5` or the conversion workflow changes. It can also be started manually from **GitHub → Actions → Convert ASL model for browser → Run workflow**.

## Deploy to Vercel

1. Import `017Mehul/asl-sign-language-recognition-ai` into Vercel.
2. Set **Root Directory** to `frontend`.
3. Use the framework preset **Other**; leave build and output commands empty so Vercel serves the static HTML and model files.
4. Deploy after the conversion workflow succeeds and the generated files appear under `frontend/models/asl_tfjs/`.
5. Open the HTTPS deployment, allow camera permission, and test multiple letters.

Camera access requires HTTPS or localhost. The first visit downloads the model and MediaPipe's hand-landmarker asset. After that, inference is performed locally in the browser; sentence text is stored in that browser's local storage, and speech uses the device's browser speech synthesis.

## Important model compatibility note

The conversion workflow verifies that the source labels are A–Z and creates TensorFlow.js files, but it cannot guarantee identical accuracy across the Python MediaPipe and browser MediaPipe runtimes. Validate predictions with a real webcam after deployment. This project classifies static alphabet hand shapes; it is not continuous ASL sentence translation, and motion-based J/Z may be less reliable.
