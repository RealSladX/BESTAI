from ultralytics import YOLO

model = YOLO('ASLAlphabet.pt')

results = model(source=0, show=True, conf = 0.3, save=True)