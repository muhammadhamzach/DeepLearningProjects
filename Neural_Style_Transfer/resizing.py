from PIL import Image

def resizing(content_image_path, style_image_path):
    con_img = Image.open(content_image_path) 
    sty_img = Image.open(style_image_path) 
    sty_img = sty_img.resize((con_img.size[0], con_img.size[1]), Image.ANTIALIAS)
    sty_img.save('resized_style_image.png')