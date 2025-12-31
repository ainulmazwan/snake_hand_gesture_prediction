import streamlit as st
import pandas as pd
import plotly.express as px
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

st.set_page_config('Snake Grid test')

rows = 10
cols = 25


st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

GRID_PLACEHOLDER = st.empty()

model = joblib.load('./models/model3.joblib')

if 'snake' not in st.session_state:
        st.session_state.snake = [(5,5), (5,4), (5,3)]  # the "body parts" of the snake
        st.session_state.food = (2,7)
        st.session_state.last_move = time.time()




direction = "RIGHT"


while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb, since cv2 shows frames in bgr
    FRAME_WINDOW.image(frame)

    

    
    if time.time() - st.session_state.last_move >= 1.0:  # 1 block per second
        st.session_state.last_move = time.time()
    
        head_x, head_y = st.session_state.snake[0]

        img_resized = cv2.resize(frame, (150, 150))

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array,axis=0)
        
        img_array = preprocess_input(img_array)

        # dict_keys(['fist', 'left', 'palm', 'right'])
        class_names = ['UP', 'LEFT', 'DOWN', 'RIGHT']
        pred  = model.predict(img_array)
    
        # pred will look like [0.1, 0.2, 0.6, 0.2]
        pred_idx = np.argmax(pred[0])            # index of highest probability
        direction = class_names[pred_idx]             # class name
        conf = pred[0][pred_idx] * 100 

        

        if direction == "UP": 
            new_head = ((head_x+1) % rows, (head_y) % cols)  # % ensures that it doesnt go over the col/row limit, wrapping back to the start
        elif direction == "DOWN":  
            new_head = ((head_x-1) % rows, (head_y) % cols)
        elif direction == "RIGHT":  
            new_head = ((head_x) % rows, (head_y+1) % cols)
        elif direction == "LEFT":  
            new_head = ((head_x) % rows, (head_y-1) % cols)
        
        
        st.session_state.snake.insert(0, new_head)
        st.session_state.snake.pop()

        
    grid = [["⬜" for p in range(cols)] for p in range(rows)]
    for x, y in st.session_state.snake:
        grid[x][y] = "🟩"
    
    fx, fy = st.session_state.food
    grid[fx][fy] = "🍎"

    GRID_PLACEHOLDER.text("\n".join([" ".join(row) for row in grid]))
        
    time.sleep(0.1)
    
st.rerun()