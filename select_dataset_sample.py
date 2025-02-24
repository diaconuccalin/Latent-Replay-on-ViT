import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.vit_lr.ViTLR_model import ViTLR
from models.vit_lr.vit_lr_utils import bordering_resize, vit_lr_image_preprocessing

N_SAMPLES_FROM_EACH = 6


def main():
    # Set random seed
    random.seed(42)

    # Randomly select samples
    if os.path.exists("selected_samples.pth"):
        selected_samples = torch.load("selected_samples.pth")
    else:
        selected_samples = dict()
        for s in [1, 2, 4, 5, 6, 8, 9, 11]:
            selected_samples[s] = dict()
            for o in range(1, 51):
                selected_samples[s][o] = random.sample(
                    range(0, 300), N_SAMPLES_FROM_EACH
                )

        # Save samples to file
        torch.save(selected_samples, "selected_samples.pth")

    # Weights path
    weights_path = "weights/trained_models/_best_native_ar1_star_rms_3000/final.pth"

    # Prepare model
    model = ViTLR(
        device="cuda",
        num_blocks=12,
        input_size=(384, 384),
        num_classes=50,
        dropout_rate=0.0,
    )

    if weights_path is not None:
        # Load stored information
        print("Loading pretrained model...")
        weights = torch.load(weights_path, weights_only=False, map_location="cuda")

        if "model_state_dict" in weights.keys():
            weights = weights["model_state_dict"]

        model.load_state_dict(weights)
        model.to("cuda")

    # Iterate through calibration images
    root_path = os.path.join("datasets", "core50", "data", "core50_350x350")
    for sequence in tqdm(selected_samples.keys()):
        sequence_path = os.path.join(root_path, f"s{sequence}")

        for obiect in selected_samples[sequence].keys():
            model.transformer.current_object = obiect
            object_path = os.path.join(sequence_path, f"o{obiect}")

            for sample in selected_samples[sequence][obiect]:
                sample_path = os.path.join(
                    object_path,
                    f"C_0{sequence}_0{obiect}_" + str(sample).zfill(3) + ".png",
                )

                # Load image
                x = np.zeros(
                    (
                        1,
                        350,
                        350,
                        3,
                    ),
                    dtype=np.uint8,
                )
                x[0] = np.array(Image.open(sample_path))
                image = x.astype(np.uint8)

                # Resize image
                image = bordering_resize(
                    image,
                    input_image_size=(384, 384),
                    original_image_size=(350, 350),
                )
                x_train = vit_lr_image_preprocessing(x=(True, image), device="cuda")

                # Forward pass
                with torch.no_grad():
                    test = model(x_train)
                    print(torch.argmax(test).item() + 1, obiect)
                    input("Press Enter to continue...")

    return None


if __name__ == "__main__":
    main()
