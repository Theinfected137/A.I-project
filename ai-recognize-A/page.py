import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import requests
import io

# -------------------
# CSS para fundo preto, texto branco e botões totalmente brancos
# -------------------
st.markdown(
    """
    <style>
    /* Fundo da página e texto */
    .stApp {
        background-color: black;
        color: white;
    }
    /* Títulos e textos */
    .css-1d391kg, .css-1adrfps {
        color: white;
    }
    /* Botões totalmente brancos */
    div.stButton > button {
        background-color: white;
        color: black;
        border: 2px solid white;  /* contorno branco */
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Draw a Character")

# -------------------
# Canvas com fundo preto e cor da linha branca
# -------------------
canvas = st_canvas(
    fill_color="black",     
    stroke_width=10,
    stroke_color="white",   
    width=200,
    height=200,
)

# -------------------
# Botão Predict
# -------------------
if st.button("Predict"):
    if canvas.image_data is not None:
        # Pega imagem do canvas e converte para grayscale
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        img = ImageOps.invert(img)  # EMNIST espera letra branca no fundo preto

        # -------------------
        # Normalização e centralização
        # -------------------
        # Rotacionar automaticamente para "cima"
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Cortar espaços em branco (bounding box)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Redimensionar mantendo proporção
        img.thumbnail((28, 28), Image.Resampling.LANCZOS)

        # Fundo preto 28x28 e colar desenho centralizado
        new_img = Image.new("L", (28, 28), 0)
        left = (28 - img.width) // 2
        top = (28 - img.height) // 2
        new_img.paste(img, (left, top))
        img = new_img

        # -------------------
        # Enviar para a API
        # -------------------
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"image": buf}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        if response.status_code == 200:
            st.write(f"Predicted class: {response.json()['class']}")
        else:
            st.error("Erro ao conectar com a API.")
