import streamlit as st
import pandas as pd
import plotly.express as px
import time
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import random

st.set_page_config('Snakey hand')

rows = 10
cols = 10


st.title("Hand Gesture Controlled Snake Game")

run = st.checkbox('Run')

left_col, right_col = st.columns(2)

with left_col: 
    FRAME_WINDOW = st.image([])
    PRED_PLACEHOLDER = st.empty()
    DIR_PLACEHOLDER = st.empty()
    CONF_PLACEHOLDER = st.empty()

with right_col:
    # st.subheader("Snake Game")
    SUBTITLE_PLACEHOLDER = st.empty()
    SCORE_PLACEHOLDER = st.empty()
    GRID_PLACEHOLDER = st.empty()
    
camera = cv2.VideoCapture(0)


model = joblib.load('../models/model3.joblib')

if 'snake' not in st.session_state:
        st.session_state.snake = [(5,5), (5,4), (5,3)]  # initializing the "body parts" of the snake
        st.session_state.food = (2,7)  # initializing the position of food
        st.session_state.last_move = time.time()




direction = "RIGHT"
st.session_state.direction = direction #initializing direction
st.session_state.label = "wait.."

st.session_state.score = 0 # initializing score


while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb, since cv2 shows frames in bgr
    FRAME_WINDOW.image(frame)

    SUBTITLE_PLACEHOLDER.markdown("Snake Game")
    SCORE_PLACEHOLDER.markdown(f"Score: {st.session_state.score}")

    

    
    if time.time() - st.session_state.last_move >= 1.0:  # 1 block per second
        st.session_state.last_move = time.time()
    
        head_x, head_y = st.session_state.snake[0]

        img_resized = cv2.resize(frame, (150, 150))

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array,axis=0)
        
        img_array = preprocess_input(img_array)

        # dict_keys(['fist', 'left', 'palm', 'right'])
        direction_names = ['DOWN', 'LEFT', 'UP', 'RIGHT']
        pred  = model.predict(img_array)
    
        # pred will look like [0.1, 0.2, 0.6, 0.2]
        pred_idx = np.argmax(pred[0])            # index of highest probability
        change_to = direction_names[pred_idx]             # class name
        conf = pred[0][pred_idx] * 100 


        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
        
        if direction == 'UP':
            # rows in a grid grows downwards, so if i want to go up, row coordinate - 1
            new_head = ((head_x-1) % rows, (head_y) % cols)  # % ensures that it doesnt go over the col/row limit, wrapping back to the start
        if direction == 'DOWN':
            new_head = ((head_x+1) % rows, (head_y) % cols)
        if direction == 'LEFT':
            new_head = ((head_x) % rows, (head_y-1) % cols)
        if direction == 'RIGHT':
            new_head = ((head_x) % rows, (head_y+1) % cols)
              
        
        
        
        st.session_state.snake.insert(0, new_head)
        # check if food eaten
        if new_head == st.session_state.food:
            # generate new food position
            st.session_state.food = (
                random.randrange(0, rows),
                random.randrange(0, cols)
            )
            st.session_state.score += 1
        else:
            # remove tail if no food eaten
            st.session_state.snake.pop()

        class_names =  ['fist', 'left', 'palm', 'right']
        # dict_keys(['fist', 'left', 'palm', 'right'])
        st.session_state.label = class_names[pred_idx] 

        st.session_state.direction = direction

        PRED_PLACEHOLDER.markdown(
            f"Prediction: {st.session_state.label}"
        )
        
        DIR_PLACEHOLDER.markdown(
            f"Direction: {st.session_state.direction}"
        )
        
        CONF_PLACEHOLDER.markdown(
            f"Confidence: {conf:.1f}%"
        )
    

        
    grid = [["⬜" for p in range(cols)] for p in range(rows)]
    for x, y in st.session_state.snake:
        grid[x][y] = "🟩"
    
    fx, fy = st.session_state.food
    grid[fx][fy] = "🍎"

    GRID_PLACEHOLDER.text("\n".join([" ".join(row) for row in grid]))

        
    time.sleep(0.1)
    
st.rerun()