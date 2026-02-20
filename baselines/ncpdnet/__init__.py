from .backbones import ImageNetCNN, ImageNetUNET
__all__ = [
    "ImageNetCNN",
    "ImageNetUNET",
    "NCPDNET",
    "CrossDomainNet"
]

ImagRefinement_REGISTRY = {
    "imagenetcnn": ImageNetCNN,
    "imagenetunet": ImageNetUNET,
}

def build_model(name: str, **kwargs):
    name = name.lower()
    if name not in ImagRefinement_REGISTRY:
        raise ValueError(f"Unknown image refinement module '{name}'. Available: {list(ImagRefinement_REGISTRY.keys())}")
    return ImagRefinement_REGISTRY[name](**kwargs)

from .ncpdnet_arch import NCPDNET, CrossDomainNet