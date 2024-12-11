from model_utils import load_model, get_vision_tower
from caption_image import caption_image
from draw_bounding_box import draw_bounding_box

# Load the model
model_path = "4bit/llava-v1.5-13b-3GB"
model, tokenizer = load_model(model_path)

# Get the vision tower
vision_tower = get_vision_tower(model)
image_processor = vision_tower.image_processor

# Run image captioning and draw bounding box
image, output = caption_image('/content/10403.jpg', 'Give exact coordinates to place an apple in the image', image_processor, tokenizer, model)
coordinates = eval(output)  # Convert string output to a Python list
image_with_box = draw_bounding_box(image.copy(), coordinates)
image_with_box.show()  # Show the image with the bounding box
