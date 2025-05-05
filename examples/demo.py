from guitartechs_dataset import GuitarTECHSDataset
from torch.utils.data import DataLoader

# Create the dataset
dataset = GuitarTECHSDataset(
    root_dir='Guitar-TECHS-data',
    players=['P1'],
    content_types=['chords'],
    modalities=['micamp'],
    slice_dur=5,
    slice_overlap=1,
    label_bin_size=0.1
)

loader = DataLoader(dataset, batch_size=4)
batch = next(iter(loader))

print("\n Batched Metadata:")
for key in ['player', 'content_type', 'sample']:
    print(f"  {key}: {batch[key]}")

print("\n Slice start/end times:")
print(f"  slice_start: {batch['slice_start']}")
print(f"  slice_end:   {batch['slice_end']}")

print("\n Modalities in data:")
for mod, tensor in batch['data'].items():
    print(f"  - {mod}: shape={tensor.shape}, dtype={tensor.dtype}") #shape is (batch size, audiosize) where audiosize = slice_dur × sample_rate

print("\n Label tensor:")
print(f"  shape={batch['label'].shape}, dtype={batch['label'].dtype}")  #shape is (batch size, nstrings, nfrets+1, number of time bins), where timebins= slice_dur/label_bin_size
