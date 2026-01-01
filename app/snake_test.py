import streamlit as st
import pandas as pd
import plotly.express as px
import time
import cv2

st.set_page_config('Snake Grid test')

rows = 10
cols = 25


st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

GRID_PLACEHOLDER = st.empty()

if 'snake' not in st.session_state:
        st.session_state.snake = [(5,5), (5,4), (5,3)]  # the "body parts" of the snake
        st.session_state.food = (2,7)
        st.session_state.last_move = time.time()


while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    

    
    if time.time() - st.session_state.last_move >= 1.0:  # 1 block per second
        st.session_state.last_move = time.time()
    
        head_x, head_y = st.session_state.snake[0]
        new_head = ((head_x) % rows, (head_y + 1) % cols)
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