from PIL import Image, ImageDraw

# 8x8 board, 512px total
squares = 8
img_size = 512
sq = img_size // squares

img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
draw = ImageDraw.Draw(img)

dark = (60, 60, 60)      # dark squares
light = (220, 220, 220)  # light squares

for r in range(squares):
    for c in range(squares):
        color = dark if (r + c) % 2 else light
        draw.rectangle([c*sq, r*sq, (c+1)*sq, (r+1)*sq], fill=color)

img.save("checkerboard.png")
print("Wrote checkerboard.png")