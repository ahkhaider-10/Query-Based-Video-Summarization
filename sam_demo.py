import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

checkpoint_path = "C:/Users/User/OneDrive/Desktop/hlcv/project/Query-Based-Video-Summarization/segment-anything/sam_initial_checkpoint/sam_vit_h_4b8939.pth"
image_path = "C:/Users/User/OneDrive/Desktop/hlcv/project/Query-Based-Video-Summarization/cat.png"
output_path = "automatic_segmented_output.jpg"

print("Loading SAM model.")
sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device="cpu") 

print("Loading image.")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Creating automatic mask generator.")
mask_generator = SamAutomaticMaskGenerator(sam)

print("Generating masks automatically.")
masks = mask_generator.generate(image_rgb)
print(f"Generated {len(masks)} masks")

overlay = image.copy()
colors = np.random.randint(0, 255, (len(masks), 3))

for i, mask_dict in enumerate(masks):
    mask = mask_dict['segmentation']  
    color = colors[i].tolist()
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        colored_mask[:, :, c] = mask.astype(np.uint8) * color[c]
    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

cv2.imwrite(output_path, overlay)
print(f"Saved masked image to {output_path}")

cv2.imshow("SAM Automatic Mask Generator Output", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
