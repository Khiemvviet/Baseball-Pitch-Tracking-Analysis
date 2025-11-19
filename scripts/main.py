from ultralytics import YOLO

model = YOLO("run/best2.pt")  

results = model.predict(
    source="assets/Pitch_11.mp4",  
    save=True,                    
    show=True,                   
    conf=0.2,                    
    project="output",             
    name="annotated_video"        
)

print("Prediction finished! Check the output folder for annotated video.")