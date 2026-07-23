# Weights Folder

## Landmark model — no download needed

This project uses **MediaPipe Face Mesh** for face + 468-point landmark
detection (previously dlib). MediaPipe's landmark model ships bundled
inside the `mediapipe` pip package itself, so — unlike dlib's
`shape_predictor_68_face_landmarks.dat` — **there is nothing to manually
download here.** Once `pip install -r requirements.txt` finishes, landmark
detection works immediately.

## `drowsiness_cnn_cbam.h5` (GENERATED, not downloaded)

This is the trained CNN+CBAM model. You do not download this — it is
produced automatically the first time you run:

```bash
python train.py
```

(after you've added your dataset under `dataset/` — see
`dataset/README.md`). Training saves it here as `drowsiness_cnn_cbam.h5`,
and `best_checkpoint.h5` (the best epoch-by-validation-accuracy
checkpoint) alongside it.

## Quick checklist before running `python main.py`

- [ ] `dataset/drowsy/` and `dataset/not_drowsy/` populated with images
- [ ] Ran `python train.py` successfully at least once
- [ ] `weights/drowsiness_cnn_cbam.h5` now exists (produced by training)
