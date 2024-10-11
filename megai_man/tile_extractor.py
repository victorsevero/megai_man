from PIL import Image

TILES_DIR = "images/tiles"


def extract_tiles(image_path):
    prefix = image_path.split("/")[-1].split("-")[0]

    original_image = Image.open(image_path)

    width, height = original_image.size
    tile_size = 16
    spacing = 1
    rows = int((width + 1) / (tile_size + 1))
    cols = int((height + 1) / (tile_size + 1))

    i = 1

    for row in range(rows):
        for col in range(cols):
            left = col * (tile_size + spacing)
            top = row * (tile_size + spacing)
            right = left + tile_size
            bottom = top + tile_size

            subimage = original_image.crop((left, top, right, bottom))

            img2show = subimage.resize((30 * tile_size, 30 * tile_size))
            img2show.show()

            user_input = input(
                """Enter one of them:
                * Background (blank);
                * Wall/Ground (i);
                * Ladder (l);
                * Spike (p);
                """
            ).strip()
            img2show.close()

            match user_input:
                case "i":
                    subimage.save(f"{TILES_DIR}/{prefix}{str(i).zfill(2)}.png")
                    i += 1
                case "l":
                    subimage.save(f"{TILES_DIR}/{prefix}-l.png")
                case "p":
                    subimage.save(f"{TILES_DIR}/{prefix}-p.png")


if __name__ == "__main__":
    image_path = "{TILES_DIR}/guts-full.png"
    extract_tiles(image_path)
