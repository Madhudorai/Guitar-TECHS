import os
import zipfile
import librosa
import pretty_midi
import requests
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

AVAILABLE_PLAYERS = ['P1', 'P2', 'P3']
AVAILABLE_CONTENT = {
    'P1': ['chords', 'scales', 'singlenotes', 'techniques'],
    'P2': ['chords', 'scales', 'singlenotes', 'techniques'],
    'P3': ['music']
}
VALID_MODALITIES = ['directinput', 'micamp', 'exo', 'ego']
NUMSTRINGS = 6
NUMFRETS = 25

class GuitarTECHSDataset(Dataset):
    """
    PyTorch-compatible dataset for the Guitar-TECHS dataset
    Args:
        root_dir (str): Base directory where the dataset is stored (will be downloaded if missing).
        sr (int): Sample rate to load audio at. Default is 48000 Hz.
        players (list or 'all'): List of player IDs to include (e.g., ['P1', 'P2']) or 'all'.
        content_types (list or 'all'): Content categories to include (e.g., ['chords', 'scales']) or 'all'.
        modalities (list or 'all'): Audio/video types to load. Subset of ['directinput', 'micamp', 'exo', 'ego'] or 'all'.
        slice_dur (float): Duration of each slice in seconds. Required for slicing.
        slice_range (tuple): Alternative to slicing; fixed (start, end) time window in seconds.
        slice_overlap (float): Overlap between slices in seconds.
        label_bin_size (float): Duration of each time bin for MIDI labels in seconds (default: 0.1 #i.e 100ms).

    Returns:
        Each sample is a dictionary containing:
            - 'player': Player ID (e.g., 'P1')
            - 'content_type': Type of content (e.g., 'scales')
            - 'sample': Unique sample name
            - 'data': Dictionary of audio/video modalities with sliced tensors
            - 'label': Tensor of shape [6, 25, T] with note activations (string, fret, time_bin)
            - 'midi_path': Path to original MIDI file
            - 'slice_start' / 'slice_end': Start and end time (in seconds) of the current slice
    """
    def __init__(self,
                 root_dir='Guitar-TECHS',
                 sr=48000,
                 players=['all'],
                 content_types='all',
                 modalities='all',
                 slice_dur=None,
                 slice_range=None, 
                 slice_overlap=0.0, 
                 label_bin_size=0.1):

        if slice_dur and slice_range:
            raise ValueError("Cannot specify both slice_dur and slice_range.")
        if slice_overlap >= slice_dur:
            raise ValueError("slice_overlap must be less than slice_dur.")

        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
           self._download_and_extract_dataset()
        self.sr = sr
        self.slice_dur = slice_dur
        self.slice_range = slice_range
        self.slice_overlap = slice_overlap
        self.label_bin_size = label_bin_size
        self.num_strings = NUMSTRINGS
        self.num_frets = NUMFRETS

        self.players = AVAILABLE_PLAYERS if players in ['all', ['all']] else players
        assert all(p in AVAILABLE_PLAYERS for p in self.players), \
            f"Players must be a subset of {AVAILABLE_PLAYERS}"
        self.modalities = VALID_MODALITIES if modalities in ['all', ['all']] else modalities
        assert all(m in VALID_MODALITIES for m in self.modalities), \
            f"Modalities must be a subset of {VALID_MODALITIES}"

        self.index = []

        # Build sample index by scanning directinput files.
        for player in self.players:
            valid_contents = AVAILABLE_CONTENT[player]
            selected_contents = valid_contents if content_types in ['all', ['all']] else content_types
            for content in selected_contents:
                if content not in valid_contents:
                    print(f"Skipping content '{content}' for player '{player}' — not available in this player's dataset.")
                    continue
                # Construct the base directory. Note: folder naming uses lower-case for content.
                base_dir = self._get_base_dir(player, content)
                di_dir = os.path.join(base_dir, 'audio', 'directinput')
                if os.path.exists(di_dir):
                    for fname in os.listdir(di_dir):
                        if fname.startswith('directinput_') and fname.endswith('.wav'):
                            # The sample identifier is based on the file name.
                            sample_value = fname.replace('directinput_', '').replace('.wav', '')
                            self.index.append({
                                'player': player,
                                'content_type': content,
                                'sample': sample_value
                            })

        if self.slice_dur:
          self.expanded_index = []
          for i, sample_meta in enumerate(self.index):
              base_dir = self._get_base_dir(sample_meta['player'], sample_meta['content_type'])
              # use micamp for total length
              audio_path = os.path.join(base_dir, 'audio', 'micamp', f"micamp_{sample_meta['sample']}.wav")

              if not os.path.exists(audio_path):
                  continue

              duration = librosa.get_duration(path=audio_path)
              total_samples = int(duration * self.sr)

              # Load full audio
              y, _ = librosa.load(audio_path, sr=self.sr)

              slice_samples = int(self.slice_dur * self.sr)
              overlap_samples = int(self.slice_overlap * self.sr)
              hop_length = slice_samples - overlap_samples

              # Pad the signal
              pad_width = (slice_samples - len(y) % hop_length) % hop_length
              y_padded = np.pad(y, (0, pad_width), mode='constant')

              # Use librosa utils for slicing
              frames = librosa.util.frame(y_padded, frame_length=slice_samples, hop_length=hop_length)

              # For each frame, compute start and end time (in seconds)
              for s in range(frames.shape[1]):
                  start_sample = s * hop_length
                  start_sec = start_sample / self.sr
                  end_sec = start_sec + self.slice_dur
                  self.expanded_index.append((i, start_sec, end_sec))

    def _download_and_extract_dataset(self):
        """
        Downloads and extracts multiple Guitar-TECHS zip files into the root directory.
        """
        print(f"{self.root_dir} not found. Downloading dataset parts...")

        os.makedirs(self.root_dir, exist_ok=True)

        files_to_download = [
            ("P1_chords.zip", "https://zenodo.org/records/14963133/files/P1_chords.zip?download=1"),
            ("P1_scales.zip", "https://zenodo.org/records/14963133/files/P1_scales.zip?download=1"),
            ("P1_singlenotes.zip", "https://zenodo.org/records/14963133/files/P1_singlenotes.zip?download=1"),
            ("P1_techniques.zip", "https://zenodo.org/records/14963133/files/P1_techniques.zip?download=1"),
            ("P2_chords.zip", "https://zenodo.org/records/14963133/files/P2_chords.zip?download=1"),
            ("P2_scales.zip", "https://zenodo.org/records/14963133/files/P2_scales.zip?download=1"),
            ("P2_singlenotes.zip", "https://zenodo.org/records/14963133/files/P2_singlenotes.zip?download=1"),
            ("P2_techniques.zip", "https://zenodo.org/records/14963133/files/P2_techniques.zip?download=1"),
            ("P3_music.zip", "https://zenodo.org/records/14963133/files/P3_music.zip?download=1"),
        ]

        for filename, url in files_to_download:
            zip_path = os.path.join(self.root_dir, filename)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('Content-Length', 0))
                block_size = 8192

                with open(zip_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f"Downloading {filename}",
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)

            os.remove(zip_path)

        self._extract_nested_zip(self.root_dir)
        print("All parts downloaded and extracted successfully.")

    def _extract_nested_zip(self, root_dir):
        """
        Recursively extracts all zip files found within the directory tree starting at root_dir.
        After extraction, the original zip files are removed.
        """
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.zip'):
                    zip_path = os.path.join(foldername, filename)
                    extract_path = os.path.splitext(zip_path)[0]  # Folder name without .zip
                    print("Extracting:", zip_path, "to", extract_path)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    os.remove(zip_path)

    def __len__(self):
        return len(self.expanded_index)

    def _get_base_dir(self, player, content):
        """
        Constructs the base directory for a given player and content type.
        Expected naming: "_", and inside that folder may be a nested folder of the same name.
        """
        dir_name = f"{player}_{content.lower()}"
        candidate = os.path.join(self.root_dir, dir_name)
        nested = os.path.join(candidate, dir_name)
        return nested if os.path.exists(nested) else candidate

    def _get_midi_path(self, item):
        base_dir = self._get_base_dir(item['player'], item['content_type'])
        return os.path.join(base_dir, 'midi', f"midi_{item['sample']}.mid")

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr)
        return audio

    def slice_audio(self, audio, start, end):
        """
        Returns a slice of the audio corresponding to [start, end) seconds. When using slice_dur,
        if the extracted segment is shorter than the desired slice length (i.e. (end - start) * sr),
        it is padded with zeros at the end.
        """
        start_sample = int(start * self.sr)
        # Determine desired slice length in samples.
        desired_length = int(self.slice_dur * self.sr) if self.slice_dur else int((end - start) * self.sr)
        end_sample = start_sample + desired_length
        segment = audio[start_sample: min(len(audio), end_sample)]
        if len(segment) < desired_length:
            segment = np.pad(segment, (0, desired_length - len(segment)), mode='constant')
        return segment

    def parse_midi(self, midi_obj, start=None, end=None):
        """
        Converts PrettyMIDI object into a sparse tensor of shape [6, 26, T],
        where T is the number of time bins based on label_bin_size.
        Each entry is 1 if a note is active for that string/fret/time-bin.
        Fret 25 (index) is used to explicitly denote silence.
        """
        duration = end - start if (start is not None and end is not None) else midi_obj.get_end_time()
        T = int(np.ceil(duration / self.label_bin_size))
        label_tensor = torch.zeros((self.num_strings, self.num_frets + 1, T))  # +1 for silence

        active_mask = torch.zeros((self.num_strings, T), dtype=torch.bool)

        for string_index, instrument in enumerate(midi_obj.instruments):
            for note in instrument.notes:
                if start is not None and (note.end <= start or note.start >= end):
                    continue

                note_on = max(note.start, start) if start else note.start
                note_off = min(note.end, end) if end else note.end
                fret = self.pitch_to_fret(note.pitch)
                if fret is None or fret >= self.num_frets:
                    continue

                onset_bin = int((note_on - start) / self.label_bin_size) if start else int(note_on / self.label_bin_size)
                offset_bin = int((note_off - start) / self.label_bin_size) if start else int(note_off / self.label_bin_size)

                onset_bin = max(0, min(T, onset_bin))
                offset_bin = max(0, min(T, offset_bin))

                label_tensor[string_index, fret, onset_bin:offset_bin] = 1
                active_mask[string_index, onset_bin:offset_bin] = 1

        # Mark silence explicitly with fret=25 where no other fret is active
        for s in range(self.num_strings):
            silence_bins = ~active_mask[s]
            label_tensor[s, self.num_frets, silence_bins] = 1  # fret 25

        return label_tensor

    def pitch_to_fret(self, midi_note, tuning=[40, 45, 50, 55, 59, 64]):
        for string_midi in tuning[::-1]:
            fret = midi_note - string_midi
            if 0 <= fret <= 24:
                return fret
        return None

    def __getitem__(self, idx):
        real_idx, start, end = self.expanded_index[idx]
        item = self.index[real_idx]
        base_dir = self._get_base_dir(item['player'], item['content_type'])

        data = {}
        # Load each modality.
        for dtype in self.modalities:
            if dtype in ['directinput', 'micamp']:
                folder = os.path.join('audio', dtype)
                ext = '.wav'
            elif dtype in ['exo', 'ego']:
                folder = os.path.join('video', dtype)
                ext = '.mp3'
            else:
                continue

            path = os.path.join(base_dir, folder, f"{dtype}_{item['sample']}{ext}")
            if os.path.exists(path):
                modality_data = self.load_audio(path)
                # Slice (and pad if needed) the audio for the desired time window.
                audio_array = self.slice_audio(modality_data, start, end) if start is not None else modality_data
            if audio_array is not None:
                data[dtype] = torch.from_numpy(audio_array).float()  # Convert to float tensor
            else:
                data[dtype] = None  # Or a zero tensor? 


        # Process MIDI labels for the corresponding time window.
        midi_path = self._get_midi_path(item)
        if os.path.exists(midi_path):
            midi_obj = pretty_midi.PrettyMIDI(midi_path)
            label_tensor = self.parse_midi(midi_obj, start, end)
        else:
            T = int((end - start) / self.label_bin_size)
            label_tensor = torch.zeros((self.num_strings, self.num_frets, T))

        # Return the sample dictionary including sample name and slice timestamps.
        return {
            'player': item['player'],
            'content_type': item['content_type'],
            'sample': item['sample'],
            'data': data,
            'label': label_tensor,
            'midi_path': midi_path,
            'slice_start': start,
            'slice_end': end
        }
