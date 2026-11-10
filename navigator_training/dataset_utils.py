import os
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    split = "train" if is_train else "val"
    root = os.path.join(args.data_path, split)
    dataset = datasets.ImageFolder(root, transform=transform)
    if len(dataset.classes) != 2:
        raise ValueError(f"Expected two classes under {root}, found {dataset.classes}")
    print(f"Loaded {len(dataset)} images from {root}")
    return dataset, 2


def _insert_before_tensor(transform, augmentation):
    index = next(
        (i for i, item in enumerate(transform.transforms)
         if isinstance(item, transforms.ToTensor)
         or item.__class__.__name__.lower() in {"totensor", "maybetotensor", "piltotensor"}),
        len(transform.transforms),
    )
    transform.transforms.insert(index, augmentation)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        spatial = transforms.Compose([
            transforms.RandomHorizontalFlip(p=args.spatial_hflip),
            transforms.RandomVerticalFlip(p=args.spatial_vflip),
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=args.spatial_degrees,
                    translate=(args.spatial_translate, args.spatial_translate),
                    scale=(args.spatial_scale_min, args.spatial_scale_max),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=(255, 255, 255),
                )],
                p=args.spatial_affine_prob,
            ),
        ])
        appearance = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.30, contrast=0.30, saturation=0.25, hue=0.03
            )], p=0.8),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.10
            ),
        ])
        _insert_before_tensor(transform, spatial)
        _insert_before_tensor(transform, appearance)
        return transform

    return transforms.Compose([
        transforms.Resize(
            (args.input_size, args.input_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
