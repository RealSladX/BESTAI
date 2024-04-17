from ultralytics import YOLO

model = YOLO('ASLAlphabet.pt')

results = model(source=0, show=True, conf = 0.3, save=True, stream=True)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
