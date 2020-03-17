from collections import defaultdict, namedtuple
import os
import math
from process import run_parallel
from pretty_midi import PrettyMIDI
import uuid
import numpy as np
from datetime import datetime
from glob import glob
import h5py

class IrregularTimeSignatureError(Exception):
    pass

class InvalidNoteSequenceError(Exception):
    pass

NOTEOFF = 128
PAUSE = 129
Note = namedtuple('Note', ['pitch', 'end'])
PauseNote = Note(-1, -1)

OUT_DIR = f"midi-processed-{datetime.now().strftime('%m-%d_%H:%M:%S')}"
OUT = f"midi-data-{datetime.now().strftime('%m-%d_%H:%M:%S')}.hdf5"
instrument_mask = [0]
target_seq_len_bars = 32

steps_per_sec = 100
def proc(midifile, log):
    midi_name = ''.join(os.path.basename(midifile).split('.')[:-1])
    # midi_dir = os.path.join(OUT_DIR, midi_name)
    
    # if os.path.exists(midi_dir):
    #     raise FileExistsError('Output directory not empty!')
    # os.mkdir(midi_dir)
    
    midi = PrettyMIDI(midifile)
    # Check time signature
    if len(midi.time_signature_changes) != 0:
        for t in midi.time_signature_changes:
            if t.numerator != 4 or t.denominator != 4:
                raise IrregularTimeSignatureError()
    
    for instrument in midi.instruments:
        if instrument.program not in instrument_mask:
            continue
        # First quantize all notes
        q_notes = defaultdict(list) # { start: Note[] }
        last_step = -1
        for note in instrument.notes:
            q_start = int(math.round(note.start * steps_per_sec))
            q_end = int(math.round(note.end * steps_per_sec))
            # Make sure that start and end aren't on the same step
            if q_start == q_end:
                q_end += 1
            q_notes[q_start].append(Note(note.pitch, q_end))
            # Find last step
            if q_end > last_step:
                last_step = q_end
        # Then go over each step, and find the highest pitch for each
        pitch_seq = []
        curr_note = PauseNote
        for step in range(last_step + 1):
            if step == curr_note.end:
                    pitch_seq.append(NOTEOFF)
                    curr_note = PauseNote
            if step in q_notes:
                # Get highest note starting on this step
                new_note = max(q_notes[step], key=lambda x: x.pitch)
                if new_note.pitch > curr_note.pitch:
                    # If there's a currently playing note first stop it
                    if curr_note is not PauseNote:
                        pitch_seq.append(NOTEOFF)
                    pitch_seq.append(new_note.pitch)
                    curr_note = new_note
            else:
                # Add pause
                pitch_seq.append(PAUSE)
        # Convert to 1D ndarray
        pitch_seq = np.array(pitch_seq)
        # Trim silence
        for i in range(len(pitch_seq)):
            if pitch_seq[i] != PAUSE:
                pitch_seq = pitch_seq[i:]
                break
        # Pad to 16 bars
        rem = pitch_seq.size % 16
        if rem != 0:
            pitch_seq = np.pad(pitch_seq, (0,16-rem), constant_values=PAUSE)
        # Trim to smaller pieces
        target_seq_len = target_seq_len_bars*16
        cut_tracks = []
        is_open = False
        new_track = []
        for note in pitch_seq:
            if note == NOTEOFF:
                if not is_open:
                    raise InvalidNoteSequenceError()
                else:
                    is_open = False
            elif note != PAUSE:
                is_open = True
            new_track.append(note)
            if len(new_track) % 16 == 0 and len(new_track) >= target_seq_len and not is_open:
                cut_tracks.append(np.array(new_track))
                new_track = []
        if len(new_track) != 0 and len(new_track) % 16 == 0:
            cut_tracks.append(np.array(new_track))
        # Save it to disk
        track_file = os.path.join(OUT_DIR, f'{midi_name}_{instrument.program}.npz')
        np.savez(track_file, *cut_tracks)

def concat(files):
    len_d = defaultdict(lambda: [[]])
    target_size = 10_000
    for idx, file in enumerate(files):
        npz = np.load(file)
        for f in npz.files:
            arr = npz[f]
            l = arr.shape[0]
            if len(len_d[l][-1])*l >= target_size:
                len_d[l].append([])
            len_d[l][-1].append(arr)

    f = h5py.File(OUT, 'a')
    g = f.create_group('data')
    counter = 0
    for batch_ls in len_d.values():
        for batch in batch_ls:
            arr = np.array(batch)
            g.create_dataset(str(counter), data=arr)
            counter += 1

if __name__ == "__main__":
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    run_parallel(proc, 'test')
    files = glob(f'{OUT_DIR}/*.npz')
    concat(files)