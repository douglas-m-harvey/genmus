import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.io.wavfile import write


musical_alphabet = ["C", "C#/Db", "D", "Eb/D#", "E", "F", "F#/Gb", "G", "Ab/G#", "A", "Bb/A#", "B"]
frequencies = [440*(2**((m - 69)/12)) for m in range(12, 120)]
notes_lut = {}
octave = -1
for index, frequency in enumerate(frequencies):
    index %= 12
    if index == 0:
        octave +=1
    note = musical_alphabet[index] + str(octave)
    notes_lut[note] = frequency
notes_full_arr = np.array(list(notes_lut.keys()))

def semitone_shift(note, semitones):
    if note is not None:
        return notes_full_arr[np.where(notes_full_arr == note)[0] + semitones][0]
    else:
        return None

def harmonics_calculator(base_frequncy, no_harmonics):
    harmonics = []
    for harmonic in range(1, no_harmonics + 1):
        harmonic_frequency = base_frequncy*harmonic
        frequency_difference = np.inf
        for note, note_frequency in notes_lut.items():
            if abs(harmonic_frequency - note_frequency) < frequency_difference:
                frequency_difference = abs(harmonic_frequency - note_frequency)
                harmonic_note = note
        harmonics.append((harmonic_note, harmonic_frequency, notes_lut[harmonic_note]))
    return harmonics

def generate_pitch(note, duration, sampling_rate = 44100):
    if note is not None:
        return np.sin(2*np.pi*notes_lut[note]*np.linspace(0, duration, int(sampling_rate*duration)))
    else:
        return np.zeros((int(sampling_rate*duration)))
    
def wobbly_piano(note_0, note_1, duration, filter_width, sample_rate):
    if note_0 is not None and note_1 is not None:
        p_0 = generate_pitch(note_0, duration, sample_rate) + generate_pitch(harmonics_calculator(notes_lut[note_0], 5)[-1][0], duration, sample_rate) + generate_pitch(harmonics_calculator(notes_lut[note_0], 4)[-1][0], duration, sample_rate)
        p_1 = generate_pitch(note_1, duration, sample_rate) + generate_pitch(harmonics_calculator(notes_lut[note_1], 3)[-1][0], duration, sample_rate) + generate_pitch(harmonics_calculator(notes_lut[note_1], 2)[-1][0], duration, sample_rate)
        e = np.exp(np.linspace(1, -1, p_0.size))
        return convolve((p_0*p_1*e)**3, ((1 - np.cos(np.linspace(0, 2*np.pi, filter_width)))/2)**3, mode = "same")
    else:
        return np.zeros(int(duration*sample_rate))

def adsr(a = 1, d = 1, s = 1, r = 1, s_level = 0.5, duration = 1, amplitude = 1, sampling_rate = 44100):
    if duration != 0:
        total = np.sum([a, d, s, r])
        a /= total
        d /= total
        s /= total
        r /= total
        attack = gaussian_filter(np.linspace(0, 1, int(sampling_rate*duration*a)), 1000)
        attack -= attack.min()
        attack /= attack.max()
        decay = gaussian_filter(np.linspace(1, 0, int(sampling_rate*duration*d)), 1000)
        decay -= decay.min()
        decay /= decay.max()/(1 - s_level)
        decay += s_level
        sustain = np.full(int(sampling_rate*duration*s), 1, dtype = float)
        sustain *= s_level
        release = gaussian_filter(np.linspace(1, 0, int(sampling_rate*duration*r)), 1000)
        release -= release.min()
        release /= release.max()/s_level
        envelope = np.concatenate([attack, decay, sustain, release])
        if envelope.shape[0] != int(duration*sampling_rate):
            envelope = np.concatenate([envelope, np.zeros((np.abs(int(duration*sampling_rate) - envelope.shape[0])))])
        return envelope*amplitude
    else:
        return np.zeros(1)

def convert_bars(bars_input, note_values_input, bpm = 90, time_signature = (4, 4), semitones = 0):
    for note_values_index, note_values in enumerate(note_values_input):
        if sum(note_values) < time_signature[0]/time_signature[1]:
            note_values.append(time_signature[0]/time_signature[1] - sum(note_values))
            bars_input[note_values_index].append(None)
        elif sum(note_values) > time_signature[0]/time_signature[1]:
            raise Exception("Too many notes in the bar!")
    bars = np.full((len(bars_input), max([len(l) for l in bars_input])), None)
    for bar_index, bar in enumerate(bars_input):
        bars[bar_index, :len(bar)] = bar
    note_values = np.full_like(bars, 0, dtype = float)
    for note_value_index, note_value in enumerate(note_values_input):
        note_values[note_value_index, :len(note_value)] = note_value
    note_durations = note_values*time_signature[1]*60/bpm
    if semitones != 0:
        for index in np.ndindex(bars.shape):
            bars[index] = semitone_shift(bars[index], semitones)
    return bars, note_durations

def generate_wave(bars, note_durations, envelope_params, sampling_rate):
    total_samples = int(sampling_rate*note_durations.sum())
    wave = np.concatenate([generate_pitch(note, note_durations[index], sampling_rate)*adsr(*envelope_params, note_durations[index], 1, sampling_rate) for index, note in np.ndenumerate(bars)])
    if wave.size < total_samples:
        wave = np.concatenate([wave, np.zeros((np.abs(wave.size - total_samples)))])
    elif wave.size > total_samples:
        wave = wave[:-np.abs(wave.size - total_samples)]
    return wave

def save(file_name, array, sample_rate):
    array -= array.min()
    array *= (2**32 - 1)/array.max()
    array -= (2**32)//2
    array = array.astype(np.int32).T
    if file_name[-4:] != ".wav":
        file_name += ".wav"
    write(file_name, sample_rate, array)