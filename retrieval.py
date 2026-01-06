import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from config import cfg
import argparse
from model.make_model_clipreid import make_model
import logging
from utils.logger import setup_logger


def preprocess_image(image_path, cfg):
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # shape: [1, 3, H, W]


def extract_feature(model, img_path, cfg):
    img_tensor = preprocess_image(img_path, cfg).cuda()
    with torch.no_grad():
        feat = model(x=img_tensor, get_image=True)
        feat = torch.nn.functional.normalize(feat, dim=1)
    return feat


def main():
    parser = argparse.ArgumentParser(description="Image similarity search using CLIP-ReID")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", type=str)
    parser.add_argument("--image_path", required=True, help="Path to query image")
    parser.add_argument("--train_folder", required=True, help="Path to training images (used as gallery)")
    parser.add_argument("--output_dir", default="./top_similar", help="Path to save top similar images")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("clipreid-test", cfg.OUTPUT_DIR, if_train=False)
    logger.setLevel(logging.WARN)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata (for model shape)
    from datasets.make_dataloader_clipreid import make_dataloader
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Build and load model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param("/home/nirmala/person_search/real-time/CLIP-ReID-master/real_time_output/ViT-B-16_60.pth")
    model.eval()
    model.cuda()

    # ---- Extract query feature ----
    query_feat = extract_feature(model, args.image_path, cfg)

    # ---- Extract gallery features from train folder ----
    gallery_feats = []
    gallery_paths = []
    for root, _, files in os.walk(args.train_folder):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.png')):
                path = os.path.join(root, fname)
                feat = extract_feature(model, path, cfg)
                gallery_feats.append(feat)
                gallery_paths.append(path)

    gallery_feats = torch.cat(gallery_feats, dim=0)
    print(f"Extracted {len(gallery_paths)} gallery features.")

    # ---- Compute similarity ----
    similarity = query_feat @ gallery_feats.t()
    top_scores, top_indices = similarity.topk(10, dim=1)

    # ---- Save top-10 similar images ----
    print(f"\nSaving top-10 similar images to: {args.output_dir}\n")
    for rank, (score, idx) in enumerate(zip(top_scores[0], top_indices[0]), 1):
        src_path = gallery_paths[idx]
        ext = os.path.splitext(src_path)[1]
        dst_name = f"rank{rank:02d}_score{score.item():.4f}{ext}"
        dst_path = os.path.join(args.output_dir, dst_name)
        shutil.copy(src_path, dst_path)
        print(f"Rank {rank:02d}: {src_path} — similarity {score.item():.4f}")


if __name__ == "__main__":
    main()
