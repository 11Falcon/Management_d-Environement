import tkinter as tk
from tkinter import Label,Canvas
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import Markdown
from PIL import Image
import pathlib
import time
import textwrap
import PIL.Image

genai.configure(api_key='AIzaSyD0phHq4PckAy8vLYlzkAnaHy4tdwT7rxA')

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        temperature=0.5)

instruction = """
Tu es un surveillant ecologique d'une salle de classe d'informatique.
Ton rôle est de te baser sur les images de la classe pour détecter les problèmes environnementaux instantannee sans donner de solutions et sans expliquer.
Les problèmes surtout liés à la consommation d'énergie et à la presence de dechets dans la salle.
Tu detecte  juste les probleme lies aux déchets, la consommation d'énergie, Ne parle pas des probleme concernant la climatisation ni de la poubelle,Parle uniquement des probleme que tu vois reellement
"""

def response_env(img,instruction:str):
    img_model = genai.GenerativeModel('gemini-pro-vision')
    message = [instruction,img]
    response = img_model.generate_content(message)
    # response.resolve()
    print(response.text)
    return response.text





# Load a model
model = YOLO('./run/detect/train/weights/best.pt')  # load a custom model

# Define the polygonal zones
ash = (1200, 1920)
zone_1 = np.array([[0, ash[0] * 0.3], [0, ash[0]], [ash[1], ash[0]], [100, ash[0] * 0.3]], np.int32)
zone_1 = zone_1.reshape((-1, 1, 2))
zone_2 = np.array([[100, ash[0] * 0.3], [ash[1], ash[0]], [ash[1], ash[0] * 0.5], [ash[1] * 0.5, ash[0] * 0.3]], np.int32)
zone_2 = zone_2.reshape((-1, 1, 2))

cap = cv.VideoCapture(r'./WIN_20231214_18_39_07_Pro.mp4')  # specify the path to your video file

# Create a Tkinter window
root = tk.Tk()
root.title("Person Counter")

# Create frames for video and counts
# video_frame = tk.Frame(root, width=0.7 * ash[1], height=ash[0])
count_frame = tk.Frame(root, width=ash[1], height=ash[0])
# video_frame.pack(side=tk.LEFT)
count_frame.pack()

# Create labels to display counts
label_zone_1 = Label(count_frame, text="Zone 1: 0", font=("Helvetica", 10))
label_zone_2 = Label(count_frame, text="Zone 2: 0", font=("Helvetica", 10))
label_zone_3=Label(count_frame, text="Message:", font=("Helvetica", 15))
label_zone_4=Label(count_frame, text="Message:", font=("Helvetica", 15))
label_zone_5=Label(count_frame, text="Instructions", font=("Helvetica", 20))
label_zone_6=Label(count_frame, text=" ", font=("Helvetica", 10))
label_zone_1.pack(pady=7)
label_zone_2.pack(pady=7)
label_zone_3.pack(pady=7)
light_canvas1 = Canvas(count_frame, width=50, height=50)
light_canvas1.pack(pady=10)
label_zone_4.pack(pady=10)
light_canvas2 = Canvas(count_frame, width=50, height=50)
light_canvas2.pack(pady=10)
label_zone_5.pack(pady=20)
label_zone_6.pack(pady=10)

# Function to update the light indicator
def update_light(message1,message2):
    light_canvas1.delete("all")  # Clear the canvas
    light_canvas2.delete("all")  # Clear the canvas
    if message1 == "Lumiere zone 1 allumée":
        light_canvas1.create_oval(10, 10, 40, 40, fill="green")
    elif message1 == "Lumiere zone 1 éteinte":
        light_canvas1.create_oval(10, 10, 40, 40, fill="red")

    if message2 == "Lumiere zone 2 allumée":
        light_canvas2.create_oval(10, 10, 40, 40, fill="green")
    elif message2 == "Lumiere zone 2 éteinte":
        light_canvas2.create_oval(10, 10, 40,40, fill="red")

def update_count_labels(zone1_count, zone2_count,message1,message2):
    label_zone_1.config(text=f"Zone 1: {zone1_count}")
    label_zone_2.config(text=f"Zone 2: {zone2_count}")
    label_zone_3.config(text=f"Message: {message1}")
    label_zone_4.config(text=f"Message: {message2}")

# Function to update video display and counts
last_update_time = time.time()
def update_video():
    global last_update_time
    ret, img = cap.read()
    if ret:
        results = model(img)
        count_zone_1 = 0
        count_zone_2 = 0
        # Draw the polygonal zones on the image
        cv.polylines(img, [zone_1], isClosed=True, color=(255, 0, 0), thickness=2)
        cv.polylines(img, [zone_2], isClosed=True, color=(255, 255, 0), thickness=2)

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes.xyxy:
                    cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    lower_point1_inside_zone_1 = cv.pointPolygonTest(zone_1, (int(box[0]), int(box[3])), False) >= 0
                    lower_point2_inside_zone_1 = cv.pointPolygonTest(zone_1, (int(box[2]), int(box[3])), False) >= 0
                    lower_point1_inside_zone_2 = cv.pointPolygonTest(zone_2, (int(box[0]), int(box[3])), False) >= 0
                    lower_point2_inside_zone_2 = cv.pointPolygonTest(zone_2, (int(box[2]), int(box[3])), False) >= 0

                    if lower_point1_inside_zone_1 and lower_point2_inside_zone_1:
                        count_zone_1 += 1
                        # cv.putText(img, f'Zone 1: {count_zone_1}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                   

                    if lower_point1_inside_zone_2 and lower_point2_inside_zone_2:
                        count_zone_2 += 1
                    cv.putText(img, f'Zone 1: {count_zone_1} Personnes', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    cv.putText(img, f'Zone 2: {count_zone_2} Personnes', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
        
        current_time = time.time()
        if current_time - last_update_time >= 60:  # Check if a minute has passed
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            resp = response_env(img_pil, instruction)
            last_update_time = current_time  # Update the last update time

            if resp:
                label_zone_6.config(text=f"{resp}")

        if count_zone_1 >0:
            message1='Lumiere zone 1 allumée'
        else:
            message1='Lumiere zone 1 éteinte'
           
        if count_zone_2 >0:
            message2='Lumiere zone 2 allumée'
        else:
            message2='Lumiere zone 2 éteinte'
        
        update_light(message1,message2)
           
            

        
        update_count_labels(count_zone_1, count_zone_2,message1,message2)
        cv.imshow('live', img)
    
    root.after(1, update_video)

# Start updating the video
update_video()

root.mainloop()

cap.release()
cv.destroyAllWindows()
