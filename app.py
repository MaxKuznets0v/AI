from detect import detect_faces
import PySimpleGUI as sg
from PIL import Image
import io
import os


def app():
    file_input_column = [
        [sg.Text('Выберите изображение')], [sg.In(key='-file_name-', enable_events=True),
        sg.FileBrowse('Найти', file_types=(("PNG files", "*.png"), ("JPG files", "*.jpg"),
        ("ALL files", "*")), target='-file_name-')]
    ]

    layout = [
        [sg.Column(file_input_column, justification='center')],
        [sg.Column([[sg.Image(key='-image-')]], justification='center')],
        [sg.Button('Поиск лиц', key='-detect-'), sg.SaveAs('Сохранить', target='-save-'),
         sg.InputText(key='-save-', do_not_clear=False, enable_events=True, visible=False)]
    ]

    window = sg.Window('Нахождение лиц', layout, resizable=True, finalize=True)

    while True:
        event, values = window.read()

        if event == '-file_name-':
            file_name = values['-file_name-']
            try:
                window["-image-"].update(data=load_image(file_name))
            except:
                pass
        elif event == '-detect-':
            if window["-image-"].get_size() != (2, 2):
                temp_path = os.path.join(os.getcwd(), "temp/temp.png")
                detect_faces(values['-file_name-'], temp_path)
                window["-image-"].update(data=load_image(temp_path))

        elif event == '-save-':
            if window["-image-"].get_size() != (2, 2):
                path = values['-save-']
                if path and os.path.exists("temp/temp.png"):
                    im = Image.open("temp/temp.png")
                    im.save(values['-save-'])

        elif event == sg.WIN_CLOSED:
            try:
                os.remove("temp/temp.png")
            except:
                pass
            break

    window.close()


def load_image(path, size=(600, 600)):
    image = Image.open(path)
    image.thumbnail(size)
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


if __name__ == '__main__':
    app()
