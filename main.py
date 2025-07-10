import argparse
import os

import torch
import torchvision

import numpy as np
import random

from tqdm import tqdm
from models.torch_vit_constants import vit_models

from datasets.core50.constants import (
    NI_TRAINING_BATCHES,
    NC_TRAINING_BATCHES,
    NIC_CUMULATIVE_TRAINING_BATCHES,
    CORE50_CATEGORY_NAMES,
    NI_TESTING_BATCH,
    NC_TESTING_BATCH,
    NIC_CUMULATIVE_TESTING_BATCH,
    NI_BATCH_SPECIFIC_WEIGHTS,
    NC_BATCH_SPECIFIC_WEIGHTS,
    NIC_BATCH_SPECIFIC_WEIGHTS,
    AR1_STAR_LEARNING_RATES,
    CWR_STAR_LEARNING_RATES,
    AR1_STAR_FREE_LEARNING_RATES,
    NI_POPULATE_RM_EPOCHS,
    NC_POPULATE_RM_EPOCHS,
    NIC_POPULATE_RM_EPOCHS,
    NIC_SINGLE_CUMULATIVE_TRAINING_BATCHES,
)
from datasets.imagenet.constants import imagenet_correct_classes
from evaluation.evaluation_utils import plot_confusion_matrix, plot_losses
from evaluation.vit_lr_evaluation_loop import vit_lr_evaluation_pipeline
from training.PipelineScenario import (
    PipelineScenario,
    PIPELINES_WITH_RM,
    PIPELINES_WITH_LEARNING_RATE_MODULATION,
    CWR_STAR_PIPELINES,
    AR1_STAR_PURE_PIPELINES,
    AR1_STAR_FREE_PIPELINES,
    LR_PIPELINES,
)
from training.training_utils import CONSTANT_TRAINING_PARAMETERS
from training.vit_lr_training_loops import vit_training_pipeline


def get_cuda_device():
    cuda_device = None
    if torch.cuda.is_available():
        if cuda_device is None:
            cuda_device = torch.cuda.device_count() - 1
        device = torch.device("cuda:" + str(cuda_device))
        print("DEVICE SET TO GPU " + str(cuda_device) + "!\n")
    else:
        print("DEVICE SET TO CPU!\n")
        device = torch.device("cpu")

    return device


def create_arg_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session_name",
        help="Name to be used when saving the weights.",
    )
    parser.add_argument(
        "--pipeline",
        help="The pipeline to be run. One of: vit_lr_naive_finetune, vit_lr_core50_evaluation.",
    )
    parser.add_argument(
        "--current_task",
        help="The task to be used.",
        required=False,
    )
    parser.add_argument(
        "--weights_path", help="Path to the trained model weights.", required=False
    )
    parser.add_argument(
        "--profile",
        help="Activates profiling, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--rehearsal_memory_size",
        "-rm_size",
        help="Defines the number of patterns to be stored in the rehearsal memory. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[0],
    )
    parser.add_argument(
        "--runs",
        help="Defines the runs (different shuffles of the batches) to be used. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[0],
    )
    parser.add_argument(
        "--n_blocks",
        help="Defines the number of transformer blocks to be included in the ViT model. "
        + "Defaults to 12 and accepts integer values between 1 and 12. "
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[12],
    )
    parser.add_argument(
        "--do_validation",
        "-val",
        help="Activates validation step after each epoch, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--data_loader_debug_mode",
        help="Activates the data loader debug mode, defaults to False.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--latent_replay_layers",
        "-lr_layer",
        help="Defines the index of the transformer block to be used as latent replay layer. "
        + "Defaults to -1, which is allowed in all non-AR1* [free] pipelines, that have a predetermined LR layer."
        + "More values can be passed using the format val1,val2,val3...",
        type=lambda arg: list(map(int, arg.split(","))),
        required=False,
        default=[-1],
    )

    return parser


def vit_lr_core50_evaluation(device, weights_path, category_based_split, current_task):
    # Remove current dir path from weights path
    weights_path = weights_path.replace(os.getcwd() + "/", "")

    # Choose batch
    if current_task == "ni":
        batch = NI_TESTING_BATCH
    elif current_task in ["nc", "multi-task-nc"]:
        batch = NC_TESTING_BATCH
    elif current_task in ["nic", "nicv2_391"]:
        batch = NIC_CUMULATIVE_TESTING_BATCH
    else:
        raise ValueError("Invalid task name!")

    losses, accuracy, conf_mat = vit_lr_evaluation_pipeline(
        batch=batch,
        input_image_size=(384, 384),
        current_task=current_task,
        current_run=0,
        weights_path=weights_path,
        device=device,
        category_based_split=category_based_split,
    )

    # Prepare accuracy save path
    saving_path_accuracy = weights_path.replace(".pth", "_evaluation_results.txt")
    saving_path_accuracy = saving_path_accuracy.replace(
        "weights/", "evaluation_results/"
    )

    # Prepare confusion matrix save path
    saving_path_conf_mat = weights_path.replace(".pth", "_confusion_matrix.png")
    saving_path_conf_mat = saving_path_conf_mat.replace(
        "weights/", "evaluation_results/"
    )

    # Prepare losses plot save path
    saving_path_losses_plot = weights_path.replace(".pth", "_losses.png")
    saving_path_losses_plot = saving_path_losses_plot.replace(
        "weights/", "evaluation_results/"
    )

    # Create directories if they do not exist
    path_split = saving_path_accuracy.split(os.sep)
    for idx in range(len(path_split) - 1):
        if not os.path.exists(os.sep.join(path_split[: idx + 1])):
            os.mkdir(os.sep.join(path_split[: idx + 1]))

    # Write results to files
    with open(saving_path_accuracy, "w") as f:
        f.write("OBTAINED ACCURACY: %0.3f" % (accuracy * 100) + "%\n")
    plot_confusion_matrix(
        conf_mat=conf_mat,
        labels=CORE50_CATEGORY_NAMES,
        category_based_split=category_based_split,
        save_location=saving_path_conf_mat,
    )
    plot_losses(losses, save_location=saving_path_losses_plot)

    print("Evaluation successfully completed!")


def vit_lr_train(
    device,
    session_name,
    profiling_activated,
    data_loader_debug_mode,
    current_task,
    current_scenario,
    n_blocks,
    runs,
    rehearsal_memory_size=0,
    should_validate=False,
    latent_replay_layers=None,
):
    # Choose batches
    populate_rm_batches = None
    lr_modulation_batch_specific_weights = None
    if current_task == "ni":
        batches = NI_TRAINING_BATCHES
        validation_batch = NI_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NI_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LEARNING_RATE_MODULATION:
            lr_modulation_batch_specific_weights = NI_BATCH_SPECIFIC_WEIGHTS
    elif current_task in ["nc", "multi-task-nc"]:
        batches = NC_TRAINING_BATCHES
        validation_batch = NC_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NC_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LEARNING_RATE_MODULATION:
            lr_modulation_batch_specific_weights = NC_BATCH_SPECIFIC_WEIGHTS
    elif current_task in ["nic", "nicv2_391"]:
        if current_scenario == PipelineScenario.NATIVE_CUMULATIVE:
            batches = NIC_SINGLE_CUMULATIVE_TRAINING_BATCHES
        else:
            batches = NIC_CUMULATIVE_TRAINING_BATCHES
        validation_batch = NIC_CUMULATIVE_TESTING_BATCH

        if current_scenario in PIPELINES_WITH_RM:
            populate_rm_batches = NIC_POPULATE_RM_EPOCHS
        if current_scenario in PIPELINES_WITH_LEARNING_RATE_MODULATION:
            lr_modulation_batch_specific_weights = NIC_BATCH_SPECIFIC_WEIGHTS
    else:
        raise ValueError("Invalid task name!")

    if current_scenario != PipelineScenario.NATIVE_CUMULATIVE:
        model_saving_frequency = 40
    else:
        model_saving_frequency = 1

    if current_scenario in CWR_STAR_PIPELINES:
        learning_rates = CWR_STAR_LEARNING_RATES
    elif current_scenario in AR1_STAR_PURE_PIPELINES:
        learning_rates = AR1_STAR_LEARNING_RATES
    elif current_scenario in AR1_STAR_FREE_PIPELINES:
        learning_rates = AR1_STAR_FREE_LEARNING_RATES
    else:
        learning_rates = [
            (0.01, 0.01),
        ] * len(batches)

    if current_scenario == PipelineScenario.LR_CWR_STAR:
        latent_replay_layers = [11]
    elif current_scenario in LR_PIPELINES:
        assert (
            latent_replay_layers is not None
        ), "Latent replay layer must be set for current scenario."
    else:
        latent_replay_layers = [-1]

    for current_run in runs:
        print(
            "--------------- Starting run",
            current_run,
            "---------------",
        )

        for num_blocks in n_blocks:
            print(
                "--------------- Starting training with",
                num_blocks,
                "blocks ---------------",
            )

            for lr_layer in latent_replay_layers:
                print(
                    "--------------- Starting training with latent replay layer",
                    lr_layer,
                    "---------------",
                )

                vit_training_pipeline(
                    current_scenario=current_scenario,
                    batches=batches,
                    initial_batches=[0],
                    current_task=current_task,
                    current_run=current_run,
                    num_blocks=num_blocks,
                    mini_batch_size=128,
                    epochs_per_batch=(
                        -1
                        if current_scenario is PipelineScenario.NATIVE_CUMULATIVE
                        else 1
                    ),
                    rehearsal_memory_size=rehearsal_memory_size,
                    learning_rates=learning_rates,
                    populate_rm_epochs=populate_rm_batches,
                    device=device,
                    session_name=session_name
                    + "_run_"
                    + str(current_run)
                    + "_blocks_"
                    + str(num_blocks)
                    + "_lr_layer_"
                    + str(lr_layer),
                    lr_modulation_batch_specific_weights=lr_modulation_batch_specific_weights,
                    model_saving_frequency=model_saving_frequency,
                    randomize_data_order=True,
                    category_based_split=False,
                    profiling_activated=profiling_activated,
                    data_loader_debug_mode=data_loader_debug_mode,
                    should_validate=should_validate,
                    validation_batch=validation_batch,
                    latent_replay_layer=lr_layer,
                    **CONSTANT_TRAINING_PARAMETERS,
                )


def evaluate_model(
    model,
    img_transforms,
    eval_imgs_paths,
    eval_gts,
    device,
    batch_size=64,
):
    """
    Evaluates the model on the given images and ground truths.
    Returns accuracy and confusion matrix.
    """
    model.to(device)
    model.eval()

    # Run inference on evaluation images
    n_batches = len(eval_imgs_paths) // batch_size
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(range(n_batches))
    progress_bar.set_description("Accuracy so far:... Evaluating model")
    for batch_idx in progress_bar:
        # Process evaluation images
        eval_images_tensor = []

        for img_i in range(batch_size):
            # Load and preprocess image
            img = torchvision.io.read_image(eval_imgs_paths[batch_idx * batch_size + img_i], mode="RGB").float() / 255.0
            img = img_transforms(img)
            eval_images_tensor.append(img)

        eval_images_tensor = torch.stack(eval_images_tensor).to(device)

        # Forward pass through the model
        with torch.no_grad():
            results = model(eval_images_tensor)

        # Compute accuracy
        _, predicted_labels = torch.max(results, dim=1)

        correct_predictions += (predicted_labels == torch.tensor(eval_gts[batch_idx * batch_size: (batch_idx + 1) * batch_size]).to(device)).cpu().sum().item()
        total_predictions += len(predicted_labels)

        progress_bar.set_description(
            f"Accuracy so far: {correct_predictions / total_predictions * 100:.2f}%. Evaluating model"
        )

    return correct_predictions / total_predictions


activations = {}
def get_hook(module_name, root_dir, act_dir):
    def hook_fn(module, _, output):
        # Save to file
        if module_name not in activations:
            activations[module_name] = 0

        torch.save(
            output.flatten(),
            os.path.join(
                root_dir,
                act_dir,
                module_name + "_activations_" + str(activations[module_name]) + ".pt"))
        activations[module_name] += 1
    
    return hook_fn


def block_level_pruning(
    model_name,
    device,
    dataset="imagenet",
    calibration_sample_size=200,
    batch_size=64,
    ):
    # Check correct model
    assert model_name in [
        "vit_b_16_default",
        "vit_b_16_swag_linear",
        "vit_b_16_swag_e2e",
        "vit_b_32_default",
        "vit_l_16_default",
        "vit_l_16_swag_linear",
        "vit_l_16_swag_e2e",
        "vit_l_32_default",
        "vit_h_14_swag_linear",
        "vit_h_14_swag_e2e",
    ], "Invalid model name!"

    # Prepare result directory
    if not os.path.exists(os.path.join(
        "results",
        model_name,
    )):
        os.makedirs(os.path.join(
            "results",
            model_name,
        ))

    # ---- Select random calibration dataset
    val_images = None
    val_gt = None

    if dataset == "imagenet":
        print("Preparing calibration set...")
        # Prepare paths
        scratch_root = os.environ["SCRATCH"]
        datasets_root = os.path.join(scratch_root, "datasets")
        imagenet_root = os.path.join(datasets_root, "imagenet")

        val_gt_path = os.path.join(imagenet_root, "gt_val.txt")
        val_images_path = os.path.join(imagenet_root, "val")

        # Load validation ground truth and extract labels
        with open(val_gt_path, "r") as f:
            val_gt = f.readlines()
        val_gt = [int(line.strip().split(" ")[0]) for line in val_gt]

        # Fix labels to match the correct classes
        with open(os.path.join(
            "datasets", "imagenet", "map_cls_all.txt"
        ), "r") as f:
            lines = [el.strip().split(" ") for el in f.readlines()]
        
        # Sort by old index
        lines = sorted(lines, key=lambda x: int(x[1]))

        # Update labels
        val_gt = [int(lines[int(label) - 1][3]) for label in val_gt]
        
        # Select random subset of validation images
        random_indices = np.random.choice(
            len(val_gt), size=calibration_sample_size, replace=False
        )

        val_images = sorted(os.listdir(val_images_path))
        val_images = [os.path.join(val_images_path, im) for im in val_images]
        calibration_images = [val_images[i] for i in random_indices]
        calibration_labels = [val_gt[i] for i in random_indices]
    else:
        raise ValueError("Invalid dataset name for block level pruning!")

    # ---- Generate calibration activations
    # Load pre-trained ViT model and set required variables
    print("Loading pre-trained ViT model...")

    weights = vit_models[model_name]["weights"]
    model = vit_models[model_name]["model"](weights=weights)    
    img_transforms = weights.transforms()

    # Initial evaluation
    initial_accuracy = evaluate_model(
        model=model,
        img_transforms=img_transforms,
        eval_imgs_paths=val_images,
        eval_gts=val_gt,
        device=device,
        batch_size=64,
    )

    with open(os.path.join(
        "results",
        model_name,
        "accuracy.txt",
    ), "w") as f:
        f.write(f"Initial accuracy: {initial_accuracy * 100:.2f}%\n")

    # Create forward hooks to capture activations
    print("Registering forward hooks...")
    hooks = []
    reference_name_len = len("encoder.layers.encoder_layer_")
    for name, module in model.named_modules():
        if reference_name_len < len(name) < reference_name_len + 3:
            hooks.append(
                module.register_forward_hook(
                    get_hook(
                        module_name=name,
                        root_dir=os.path.join(
                            os.environ["FAST"],
                            "activations_calin",),
                        act_dir=model_name,
            )))
    print(len(hooks), "hooks registered!")

    # Set model to evaluation mode
    model.eval()

    # Prepare directory for saving activations
    if not os.path.exists(os.path.join(
        os.environ["FAST"],
        "activations_calin",
        model_name,
    )):
        os.makedirs(os.path.join(
            os.environ["FAST"],
            "activations_calin",
            model_name,
        ))

    # Run inference on calibration images
    n_batches = len(calibration_images) // batch_size
    for batch_idx in tqdm(range(n_batches), desc="Running inferences for calibration"):
        # Process calibration images
        calibration_images_tensor = []

        for img_i in range(batch_size):
            # Load and preprocess image
            img = torchvision.io.read_image(calibration_images[img_i], mode="RGB").float() / 255.0
            img = img_transforms(img)
            calibration_images_tensor.append(img)
            del img

        calibration_images_tensor = torch.stack(calibration_images_tensor).to(device)

        # Forward pass through the model
        with torch.no_grad():
            results = model(calibration_images_tensor)
            
        model.zero_grad()
        del calibration_images_tensor

    # Compute number of blocks
    n_blocks = len(activations)

    # ---- Compute activation distribution similarity
    print("Computing activation distributions...")
    histograms = list()
    for block_idx in range(n_blocks):
        block_tensor = None

        for batch_idx in range(n_batches):
            # Load activations from files
            file_name = os.path.join(
                os.environ["FAST"],
                "activations_calin",
                model_name,
                "encoder.layers.encoder_layer_" + str(block_idx) + "_activations_" + str(batch_idx) + ".pt"
            )

            if os.path.exists(file_name):
                current_tensor = torch.load(file_name)
                
                if block_tensor is None:
                    block_tensor = current_tensor
                else:
                    block_tensor = torch.cat((block_tensor, current_tensor), dim=0)
            else:
                print(f"File {file_name} does not exist, skipping...")

        histograms.append(torch.histc(
            block_tensor,
            bins=256,
        ))

    # Compute similarity between histograms
    print("Computing histogram similarity...")
    similarity_matrix = torch.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i != j:
                similarity_matrix[i, j] = torch.cosine_similarity(
                    histograms[i].flatten(),
                    histograms[j].flatten(),
                    dim=0,
                )
            else:
                similarity_matrix[i, j] = -1.0

    # ---- Iterate over blocks, prune, and evaluate
    print("Iterating over blocks to prune and evaluate...")
    remaining_blocks = list(range(n_blocks))

    for block_idx in range(n_blocks - 1):
        # -- Prune block
        # Find block to prune
        max_similarity = 0.0
        block_to_prune = remaining_blocks[0]

        for i in range(len(remaining_blocks) - 1):
            if similarity_matrix[remaining_blocks[i], remaining_blocks[i + 1]] > max_similarity:
                max_similarity = similarity_matrix[remaining_blocks[i], remaining_blocks[i + 1]]
                block_to_prune = remaining_blocks[i + 1]

        # Prune block from model
        model.encoder.layers.pop(remaining_blocks.index(block_to_prune))
        remaining_blocks.remove(block_to_prune)

        # -- Evaluate model performance
        initial_accuracy = evaluate_model(
            model=model,
            img_transforms=img_transforms,
            eval_imgs_paths=val_images,
            eval_gts=val_gt,
            device=device,
            batch_size=64,
        )

        with open(os.path.join(
            "results",
            model_name,
            "accuracy.txt",
        ), "a") as f:
            f.write(f"Accuracy after {n_blocks - len(remaining_blocks)} blocks removed (most recently block {block_to_prune} with similarity between input and output activation distribution {max_similarity}): {initial_accuracy * 100:.2f}%\n")

        # -- Fine-tune model

        # -- Evaluate post-tuning performance

    print("Succesfully finished!")


def main():
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    session_name = args.session_name
    pipeline = args.pipeline
    weights_path = args.weights_path
    profiling_activated = args.profile
    rehearsal_memory_sizes = args.rehearsal_memory_size
    runs = args.runs
    n_blocks = args.n_blocks
    do_validation = args.do_validation
    data_loader_debug_mode = args.data_loader_debug_mode
    current_task = args.current_task
    latent_replay_layers = args.latent_replay_layers

    if pipeline != "block_level_pruning":
        assert current_task is not None, "Current task must be specified for this pipeline."

    # Check if pipeline is supported
    available_pipelines = [
        "core50_evaluation",
        "native_cumulative_train",  # PipelineScenario.NATIVE_CUMULATIVE
        "native_cwr_star_train",  # PipelineScenario.NATIVE_CWR_STAR
        "native_ar1_star_train",  # PipelineScenario.NATIVE_AR1_STAR
        "native_ar1_star_free_train",  # PipelineScenario.NATIVE_AR1_STAR_FREE
        "lr_cwr_star_train",  # PipelineScenario.LR_CWR_STAR
        "lr_ar1_star_train",  # PipelineScenario.LR_AR1_STAR
        "lr_ar1_star_free_train",  # PipelineScenario.LR_AR1_STAR_FREE
        "block_level_pruning",
    ]
    assert (
        pipeline in available_pipelines
    ), "Pipeline currently not supported. Choose one from: " + str(available_pipelines)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Get cuda device
    device = get_cuda_device()

    # Run chosen pipeline
    if pipeline == "core50_evaluation":
        vit_lr_core50_evaluation(
            device=device,
            weights_path=weights_path,
            category_based_split=False,
            current_task=current_task,
        )
    elif pipeline == "block_level_pruning":
        block_level_pruning(model_name=session_name, calibration_sample_size=128, device=device)
    elif pipeline == "native_cumulative_train":
        vit_lr_train(
            current_scenario=PipelineScenario.NATIVE_CUMULATIVE,
            device=device,
            session_name=session_name,
            profiling_activated=profiling_activated,
            data_loader_debug_mode=data_loader_debug_mode,
            current_task=current_task,
            n_blocks=n_blocks,
            runs=runs,
            should_validate=do_validation,
        )
    else:
        current_scenario = None
        if pipeline == "native_cwr_star_train":
            current_scenario = PipelineScenario.NATIVE_CWR_STAR
        elif pipeline == "native_ar1_star_train":
            current_scenario = PipelineScenario.NATIVE_AR1_STAR
        elif pipeline == "native_ar1_star_free_train":
            current_scenario = PipelineScenario.NATIVE_AR1_STAR_FREE
        elif pipeline == "lr_cwr_star_train":
            current_scenario = PipelineScenario.LR_CWR_STAR
        elif pipeline == "lr_ar1_star_train":
            current_scenario = PipelineScenario.LR_AR1_STAR
        elif pipeline == "lr_ar1_star_free_train":
            current_scenario = PipelineScenario.LR_AR1_STAR_FREE

        for rehearsal_memory_size in rehearsal_memory_sizes:
            print(
                "--------------- Starting training with rm size of",
                rehearsal_memory_size,
                "---------------",
            )

            vit_lr_train(
                current_scenario=current_scenario,
                device=device,
                session_name=session_name + "_rms_" + str(rehearsal_memory_size),
                profiling_activated=profiling_activated,
                data_loader_debug_mode=data_loader_debug_mode,
                current_task=current_task,
                rehearsal_memory_size=rehearsal_memory_size,
                runs=runs,
                n_blocks=n_blocks,
                should_validate=do_validation,
                latent_replay_layers=latent_replay_layers,
            )

    return None


if __name__ == "__main__":
    main()
