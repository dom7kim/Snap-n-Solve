# Import necessary modules
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import requests
import base64
import json
import re
import io
import os
import openai

# Import custom modules
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Global constants for the application
# Stroke width and color are used for the rectangle drawn on the canvas
STROKE_WIDTH = 3
STROKE_COLOR = "#ffffff"
MAX_IMG_SIZE = 700
API_ENDPOINT = "https://api.mathpix.com/v3/text"
APP_TITLE = "Snap 'n' Solve - Your Math Assistant"
APP_INSTRUCTION = """
Snap 'n' Solve can handle a variety of math problems, including integrals, limits, derivatives, equations, and inequalities.

To use the application, follow these steps:

1. Upload an image with a mathematical equation or expression, either typed or handwritten.
2. Draw a rectangle around the equation or  expression in the image, ensuring it's fully covered and no other text is captured.
3. If you need a derivative, check the 'Looking for a derivative?' box and select the derivative order. Leave it unchecked otherwise.
4. Click 'Select & Solve' to solve the equation or find its derivative.
"""
WEB_LINK_1 = "[Click here to find a variety of calculus problems in PNG images](https://www.math-exercises.com/limits-derivatives-integrals)"
WEB_LINK_2 = "[Click here to find a variety of simultaneous equations in PNG images](https://www.math-exercises.com/equations-and-inequalities/systems-of-linear-equations-and-inequalities)"

# Function to retrieve Mathpix API credentials from environment variables
def get_credentials():
    return os.getenv('MATHPIX_APP_ID'), os.getenv('MATHPIX_APP_KEY')

# Function to crop an image based on given coordinates
def crop_image(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

# Function to resize an image if it exceeds a maximum size
def resize_image(img, max_size=MAX_IMG_SIZE):
    if max(img.size) > max_size:
        wpercent = (max_size / float(img.size[0])) if img.size[0] > img.size[1] else (max_size / float(img.size[1]))
        hsize = int(float(img.size[1]) * float(wpercent))
        img = img.resize((max_size, hsize), Image.LANCZOS) if img.size[0] > img.size[1] else img.resize((hsize, max_size), Image.LANCZOS)
    return img

# Function to send a request to the Mathpix OCR API
def mathpix_request(base64_img, app_id, app_key):
    image_uri = "data:image/jpg;base64," + base64_img
    try:
        response = requests.post(API_ENDPOINT, data=json.dumps({"src": image_uri, "formats": ["text", "data", "html"], "data_options": {"include_asciimath": True,"include_latex": True}}), headers={"app_id": app_id, "app_key": app_key, "Content-type": "application/json"})
        response.raise_for_status()
    except requests.exceptions.RequestException:
        st.error("An error occurred while processing the image. Please try again.")
        return None
    return response

# Function to process the response from the Mathpix OCR API
def process_response(response):
    if 'error' in response.json():
        st.error("An error occurred while processing the image. Please try again.")
    elif response.json()['is_printed'] | response.json()['is_handwritten']:
        response_text = response.json()['text']
        response_latex = re.sub('\\\\(?!\\\\)', r'\\', response_text.replace("\\(", "").replace("\\)", ""))
        return response_latex
    else:
        st.error("No equation was recognized in the image. Please ensure the image contains a clear, readable equation.")
        return None

# Function to convert a text with a mathematical expression to LaTeX format
def latexify_answer(input_text):
    system_instruction = """Just follow the instruction. No mistake.
    Put plain texts into the \\text{{}} command and then convert mathematical equations into a LaTex format. 
    If your input is 'The solution to the equation ax^2 + bx + c = 0  is given by x = (-b ± sqrt(b^2 - 4ac))/(2a)',
    your output should be '\\text{{The solution to the equation }} a x^{{2}} + b x + c = 0 \\text{{ is given by }} x = \\frac{{-b \\pm \\sqrt{{b^{{2}} - 4ac}}}}{{2a}}'.
    """
    messages = [{"role": "system", "content": system_instruction},
                    {"role": "user", "content": input_text}]
    response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo-0613',
                                            temperature = 0,
                                            messages=messages) 
    result = response['choices'][0]['message']['content']

    return result

# Function to solve an equation or find its derivative using Wolfram Alpha
def solve_equation(equation, agent, derivative_order):
    st.latex(r'\mathrm{Solving:}\;' + equation)
    with st.spinner("Wait for it..."):
        if derivative_order:
            template = """Make {eq} wolfram-friendly, try finding its {order} derivative, and describe the answer in one sentence."""
            prompt = PromptTemplate(
            input_variables=['eq', 'order'],
            template = template
            )
            answer = agent.run(prompt.format(eq=equation, order=derivative_order))
        else:
            template = """Make {eq} wolfram-friendly, try solving it, and describe the answer in one sentence."""
            prompt = PromptTemplate(
            input_variables=['eq'],
            template = template,
            )
            answer = agent.run(prompt.format(eq=equation))
    
        answer_latexified = latexify_answer("'{}'".format(answer))
        st.latex(answer_latexified)

# Main function to run the application
def main():
    st.title(APP_TITLE)
    st.markdown(APP_INSTRUCTION)
    st.markdown(WEB_LINK_1)
    st.markdown(WEB_LINK_2)

    llm = OpenAI(temperature=0)
    tool_names = ["wolfram-alpha"]
    tools = load_tools(tool_names, llm=llm)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # Handle file upload and prepare the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])  
    
    if uploaded_file is not None:
        image = resize_image(Image.open(uploaded_file).convert("RGB"))
        np_image = np.array(image)
        canvas_result = st_canvas(fill_color="rgba(255, 165, 0, 0.3)", stroke_width=STROKE_WIDTH, stroke_color=STROKE_COLOR, background_image=image, drawing_mode="rect", key="canvas", width=image.width, height=image.height)

        # Process the ROI data if it exists
        if canvas_result.json_data is not None:
            rectangles = [shape for shape in canvas_result.json_data["objects"] if shape["type"] == "rect"]
            derivative_checkbox = st.checkbox('Looking for a derivative?')
            if derivative_checkbox:
                derivative_order = st.selectbox('Order of Derivative:', ['1st', '2nd', '3rd']) if derivative_checkbox else None
            else:
                derivative_order = False   
            if rectangles:
                if st.button('Select & Solve'):
                    latest_rectangle = rectangles[-1]
                    cropped_img = crop_image(np_image, int(latest_rectangle["left"]), int(latest_rectangle["top"]), int(latest_rectangle["left"]) + int(latest_rectangle["width"]), int(latest_rectangle["top"]) + int(latest_rectangle["height"]))
                    img_pil = Image.fromarray(cropped_img)
                    img_byte_arr = io.BytesIO()
                    img_pil.save(img_byte_arr, format='JPEG')
                    base64_img = base64.b64encode(img_byte_arr.getvalue()).decode()

                    # Solve the equation or find its derivative if applicable
                    app_id, app_key = get_credentials()
                    response = mathpix_request(base64_img, app_id, app_key)
                    if response is not None:
                        equation = process_response(response)
                        if equation is not None:
                            solve_equation(equation, agent, derivative_order)
    
    # Display copyright notice
    st.markdown("""
    ***
    © 2023 Dongwon Kim
    """)

# Entry point of the program
if __name__ == "__main__":
    main()
