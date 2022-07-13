import cv2
import numpy as np
import PySimpleGUI as sg
from utils import Predictor
from PIL import Image
from io import BytesIO


col = [    
    [sg.R('Draw Boundry', 1, key='BOUND', enable_events=True)],
    [sg.B('Run Inference', key='RUN')],
]

layout = [[sg.Graph(
            canvas_size=(1050, 550),
            graph_bottom_left=(0, 550),
            graph_top_right=(1050, 0),
            key="-GRAPH-",
            enable_events=True,
            background_color='white',
            drag_submits=True,
            right_click_menu=[[],['Erase item',]]
            ), sg.Col(col, key='-COL-') ],
        [sg.Text("", size=(0, 1), key='OUTPUT')]
]

window = sg.Window("Moon Composition Predictor", layout, finalize=True)
graph = window["-GRAPH-"]
dragging = False
start_point = end_point = prior_rect = None

bgid = 0


if __name__ == "__main__":
    model = Predictor()
    img_arr = cv2.cvtColor(cv2.imread("data/moon/original/lunar_original.jpeg"), cv2.COLOR_BGR2RGB)

    im = Image.open("data/moon/original/lunar_original.jpeg").resize((1050, 550))
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    graph.draw_image(data=data, location=(0, 0))

    box = []

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-GRAPH-":
            x, y = values["-GRAPH-"]
            if not dragging:
                start_point = (x, y)
                dragging = True
                drag_figures = graph.get_figures_at_location((x,y))
                lastxy = x, y

            else:
                end_point = (x, y)

            if prior_rect:
                graph.delete_figure(prior_rect)

            delta_x, delta_y = x - lastxy[0], y - lastxy[1]
            lastxy = x,y

            if None not in (start_point, end_point):
                if values['BOUND']:
                    prior_rect = graph.draw_rectangle(start_point, end_point, line_color='red')

        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            box.append([start_point, end_point])
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            prior_rect = None

        elif event == 'RUN':
            st, ed = box[-1]
            box = []

            inputs = img_arr[
                int(1100*(st[1]/550)) : int(1100*(ed[1]/550)),
                int(2100*(st[0]/1050)) : int(2100*(ed[0]/1050)),
            :]

            iron, thor, cobt = model.predict(inputs)

            window['OUTPUT'].update(value=f"Iron concentration: {np.round(float(iron), 2)} wt% \
            Thorium concentration: {np.round(float(thor), 2)} ppm \
                Cobalt concentration: {np.round(cobt, 2)} mg/kg")


            
            

            

