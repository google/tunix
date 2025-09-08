# scripts/make_dummy_vqa.py
import os, json, random
from PIL import Image, ImageDraw

def make_dir(p):
    os.makedirs(p, exist_ok=True)

def solid_bg(size=(384,384), color=(255,255,255)):
    return Image.new("RGB", size, color)

def draw_shapes(img, kind="square", count=1, color=(255,0,0), size=64):
    """Draw 'count' shapes randomly."""
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(count):
        x = random.randint(16, W-size-16)
        y = random.randint(16, H-size-16)
        if kind == "square":
            draw.rectangle([x, y, x+size, y+size], fill=color)
        elif kind == "circle":
            draw.ellipse([x, y, x+size, y+size], fill=color)
        elif kind == "triangle":
            draw.polygon([(x, y+size), (x+size//2, y), (x+size, y+size)], fill=color)
    return img

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main(root="./dummy_vqa"):
    random.seed(42)
    img_dir = os.path.join(root, "images")
    make_dir(img_dir)

    # Define a few synthetic samples: (filename, builder_fn, question, numeric answer)
    samples = [
        # TRAIN (8)
        ("img1.png",  lambda: draw_shapes(solid_bg(), "square", 1, (255,0,0)),   "How many red squares are in the image?", 1),
        ("img2.png",  lambda: draw_shapes(solid_bg(), "square", 2, (255,0,0)),   "How many red squares are in the image?", 2),
        ("img3.png",  lambda: draw_shapes(solid_bg(), "circle", 3, (0,0,255)),   "How many blue circles are in the image?", 3),
        ("img4.png",  lambda: draw_shapes(solid_bg(), "circle", 1, (0,0,255)),   "How many blue circles are in the image?", 1),
        ("img5.png",  lambda: draw_shapes(solid_bg(), "triangle", 2, (0,255,0)), "How many green triangles are there?", 2),
        ("img6.png",  lambda: draw_shapes(solid_bg(), "triangle", 3, (0,255,0)), "How many green triangles are there?", 3),
        ("img7.png",  lambda: draw_shapes(solid_bg(), "square", 3, (255,0,0)),   "Count the red squares.", 3),
        ("img8.png",  lambda: draw_shapes(solid_bg(), "circle", 2, (0,0,255)),   "Count the blue circles.", 2),

        # EVAL (4)
        ("img9.png",  lambda: draw_shapes(solid_bg(), "square", 1, (255,0,0)),   "How many red squares are in the image?", 1),
        ("img10.png", lambda: draw_shapes(solid_bg(), "circle", 2, (0,0,255)),   "How many blue circles are in the image?", 2),
        ("img11.png", lambda: draw_shapes(solid_bg(), "triangle", 1, (0,255,0)), "How many green triangles are there?", 1),
        ("img12.png", lambda: draw_shapes(solid_bg(), "square", 2, (255,0,0)),   "Count the red squares.", 2),
    ]

    # Build images & rows
    rows = []
    for fname, maker, q, ans in samples:
        img = maker()
        path = os.path.join(img_dir, fname)
        img.save(path)
        rows.append({"image": path, "question": q, "answer": str(ans)})

    # Split: first 8 -> train, last 4 -> eval
    train_rows = rows[:8]
    eval_rows  = rows[8:]

    write_jsonl(os.path.join(root, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(root, "eval.jsonl"),  eval_rows)

    print(f"Created {len(train_rows)} train and {len(eval_rows)} eval samples in {root}")
    print(f"Example row:\n{json.dumps(train_rows[0], indent=2)}")

if __name__ == "__main__":
    main()