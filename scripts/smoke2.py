"""Inspect TIAToolbox 1.6 HoVerNet API directly from source."""
import inspect
import warnings
warnings.filterwarnings("ignore")

from tiatoolbox.models.architecture.hovernet import HoVerNet

print("=== infer_batch ===")
print(f"signature: {inspect.signature(HoVerNet.infer_batch)}")
print()
print(inspect.getsource(HoVerNet.infer_batch))

print("\n" + "="*70)
print("=== postproc ===")
print(f"signature: {inspect.signature(HoVerNet.postproc)}")
print()
src = inspect.getsource(HoVerNet.postproc)
print(src[:3000])
print()
print("..." if len(src) > 3000 else "[full source above]")

print("\n" + "="*70)
print("=== preproc (might handle padding) ===")
print(f"signature: {inspect.signature(HoVerNet.preproc)}")
print(inspect.getsource(HoVerNet.preproc)[:1500])

print("\n" + "="*70)
print("=== get_instance_info ===")
print(f"signature: {inspect.signature(HoVerNet.get_instance_info)}")
