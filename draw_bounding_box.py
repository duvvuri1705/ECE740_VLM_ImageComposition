from PIL import ImageDraw

def draw_bounding_box(image, coordinates, color="red", width=3):
    # Convert fractional coordinates to pixel coordinates
    width_img, height_img = image.size
    x_min = int(coordinates[0] * width_img)
    y_min = int(coordinates[1] * height_img)
    x_max = int(coordinates[2] * width_img)
    y_max = int(coordinates[3] * height_img)

    # Draw the bounding box on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    return image
