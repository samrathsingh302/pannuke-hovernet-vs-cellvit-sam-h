"""Find which input sizes HoVer-Net 'fast' actually accepts."""
import warnings
warnings.filterwarnings("ignore")

import torch
from tiatoolbox.models.architecture.hovernet import HoVerNet

model = HoVerNet(num_input_channels=3, num_types=6, mode="fast")
sd = torch.load("./models/tiatoolbox_weights/hovernet_fast-pannuke.pth",
                map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
model = model.cuda().eval()

print("Testing input sizes for HoVer-Net 'fast' mode:")
print(f"{'input':>6} {'output':>10} {'crop':>10} {'status':>10}")
print("-" * 50)

# Test relevant sizes — multiples of 16, 32, 64
SIZES = [256, 272, 288, 320, 352, 384, 416, 448, 480, 512]

for size in SIZES:
    test = torch.randint(0, 256, (1, size, size, 3), dtype=torch.uint8)
    try:
        with torch.no_grad():
            o = HoVerNet.infer_batch(model, test, device='cuda')
        out_size = o[0].shape[1]
        crop = size - out_size
        status = "✓"
        marker = " ← perfect for 256" if out_size == 256 else (" ← can crop to 256" if out_size > 256 else "")
        print(f"{size:>6} {out_size:>10} {crop:>10} {status:>10}{marker}")
    except RuntimeError as e:
        msg = str(e)[:50]
        print(f"{size:>6} {'✗':>10} {'-':>10} {'ERROR':>10}  {msg}")
    
    torch.cuda.empty_cache()
