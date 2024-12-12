# Image Composition Using Vision Language Model (LLaVA)

## Overview

This project demonstrates the use of a Vision Language Model (LLaVA) for image composition. By combining natural language prompts and scene images, our approach enables the generation of bounding box coordinates and composite images, allowing objects to be strategically placed within a scene based on textual descriptions.

## Key Features
1. **Bounding Box Prediction**  
   Given a scene image and a text prompt, the LLaVA model predicts bounding box coordinates for object placement.
   
   **Scene Image:** A basket of fruits.
   **Text Prompt:** "Give me bounding box coordinates to place an apple in the image."
   **Output:** The bounding box coordinates indicating where the apple can be placed.

2. **Image Composition**  
   In addition to the scene image and text prompt, an object patch (e.g., an image of an apple) is provided. Using the bounding box coordinates generated by LLaVA, the object patch is placed onto the scene image to produce a composite image.

## Workflow

1. **Input:**
   - Scene image
   - Text prompt describing the desired placement or action.
   - Optional object patch for image composition.
     
2. **Processing:**
   - The LLaVA model processes the scene image and text prompt to generate bounding box coordinates.
   - Using the coordinates, the object patch is placed onto the scene image via a Python script.

3. **Output:**
   - Bounding box coordinates.
   - A composite image where the object patch is appropriately placed based on the text prompt.


## Steps to Run the Project

1. **Clone the Repository:**
      - Clone the LLaVA repository to your machine using `git clone -b v1.0 https://github.com/camenduru/LLaVA`
      - Navigate to the directory using `cd LLaVA`
        
2. **Install Required Dependencies:**
      - Ensure you have python installed and install the gradio using `pip install -q gradio .`
        
3. **Run the code:**
      - Open google colab or any IDE and start running the project. Ensure that LLaVA and the project files are in the same directory.
        
4. **Expected Output for Basic Task:**
      - The bounding box coordinates will be printed on the console.
      - The generated image with the bounding box drawn will be displayed.


## Acknowledgments 

This project is build upon the [LLaVA repository](https://github.com/camenduru/LLaVA) and uses the model and architecture for vision-language tasks.
