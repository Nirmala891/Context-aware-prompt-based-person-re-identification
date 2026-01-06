import os
import torch
from PIL import Image
from torchvision import transforms
from config import cfg
import argparse
from model.make_model_clipreid import make_model
from utils.logger import setup_logger


def preprocess_image(image_path, cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # shape: [1, 3, H, W]


def main():
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("clipreid-test", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Get metadata (camera_num and view_num not used here)
    from datasets.make_dataloader_clipreid import make_dataloader
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Build and load model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    #TEST.WEIGHT="/home/nirmala/person_search/CLIP-ReID-master/dukemtmc_output/ViT-B-16_60.pth"
    #model.load_param(cfg.TEST.WEIGHT)
    model.load_param("/home/nirmala/person_search/CLIP-ReID-master/dukemtmc_output/ViT-B-16_60.pth")
    model.eval()
    model.cuda()

    # Load and preprocess image
    image_tensor = preprocess_image(args.image_path, cfg).cuda()

    with torch.no_grad():
        # Extract image feature
        image_feat = model(x=image_tensor, get_image=True)  # shape: [1, D]

        # Get text features for all classes
        labels = torch.arange(num_classes).cuda()
        text_feat = model(get_text=True, label=labels)  # shape: [C, D]

        # Normalize and compute similarity
        image_feat = torch.nn.functional.normalize(image_feat, dim=1)
        text_feat = torch.nn.functional.normalize(text_feat, dim=1)
        similarity = image_feat @ text_feat.t()  # [1, C]

        # Get top class
        top_score, top_class = similarity.topk(1, dim=1)
        print("Predicted class index:", top_class.item())
        print("Similarity score:", top_score.item())


if __name__ == "__main__":
    main()
