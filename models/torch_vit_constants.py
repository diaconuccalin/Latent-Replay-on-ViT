import torchvision

vit_models = {
    "vit_b_16_default": {
        "model": torchvision.models.vit_b_16,
        "weights": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
        },
    "vit_b_16_swag_linear": {
        "model": torchvision.models.vit_b_16,
        "weights": torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
        },
    "vit_b_16_swag_e2e": {
        "model": torchvision.models.vit_b_16,
        "weights": torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        },
    "vit_b_32_default": {
        "model": torchvision.models.vit_b_32,
        "weights": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1,
        },
    "vit_l_16_default": {
        "model": torchvision.models.vit_l_16,
        "weights": torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1,
        },
    "vit_l_16_swag_linear": {
        "model": torchvision.models.vit_l_16,
        "weights": torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
        },
    "vit_l_16_swag_e2e": {
        "model": torchvision.models.vit_l_16,
        "weights": torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        },
    "vit_l_32_default": {
        "model": torchvision.models.vit_l_32,
        "weights": torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1,
        },
    "vit_h_14_swag_linear": {
        "model": torchvision.models.vit_h_14,
        "weights": torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1,
        },
    "vit_h_14_swag_e2e": {
        "model": torchvision.models.vit_h_14,
        "weights": torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        },
}
