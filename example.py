import music_tools as mt
import numpy as np


bpm = 120
time_signature = (4, 4)
sample_rate = 44100
repeats = 2
envelope_params = (1/4, 1/2, 1/8, 1/8, 0.6)
bars = [[["Bb/A#4", "G4", "Eb/D#4", "C4"], ["Ab/G#4", "F4", "D4"], ["G4", "Eb/D#4", "C4", "Ab/G#3"], ["F4", "D4", "Bb/A#3"]],
        [["C3", "Bb/A#2"], ["Bb/A#2", "Ab/G#2"], ["Ab/G#2", "G3"], ["G2", "C3"]],
        [["Eb/D#3", "D3"], ["D3", "C3"], ["C3", "Bb/A#2"], ["Bb/A#2", "Eb/D#3"]],
        [["G3", "F3"], ["F3", "Eb/D#3"], ["Eb/D#3", "D3"], ["D3", "G3"]]]
note_values = [[[1/4, 1/8, 1/8, 1/2], [1/2, 1/4, 1/4], [1/4, 1/8, 1/8, 1/2], [1/4, 1/4, 1/2]],
               [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [1/2, 1/2]],
               [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [1/2, 1/2]],
               [[1/2, 1/2], [1/2, 1/2], [1/2, 1/2], [1/2, 1/2]]]
waves = np.array([mt.generate_wave(*mt.convert_bars(bars[i], note_values[i], bpm, time_signature), envelope_params, sample_rate) for i in range(len(bars))])
waves = np.concatenate([waves for _ in range(repeats)], axis = 1)
mt.save("example", waves, sample_rate)