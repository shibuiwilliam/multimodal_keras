import os
import numpy as np
import librosa


# load a wave data
def load_wave_data(filepath):
    x, _ = librosa.load(filepath, sr=44100)
    return x, _


class AudioGenerator():
    def __init__(self, melsp=True, augment=True, **params):
        self.melsp = melsp
        self.melsp_n_fft = params[
            "melsp_n_fft"] if "melsp_n_fft" in params else 1024
        self.melsp_hop_length = params[
            "melsp_hop_length"] if "melsp_hop_length" in params else 128
        self.melsp_n_mels = params[
            "melsp_n_mels"] if "melsp_n_mels" in params else 128

        self.augment = augment
        self.white_noise_minmax = params[
            "add_white_noise_rate"] if "add_white_noise_rate" in params else (
                1, 50)
        self.shift_sound_minmax = params[
            "shift_sound_rate"] if "shift_sound_rate" in params else (2, 6)
        self.stretch_sound_minmax = params[
            "stretch_sound_rate"] if "stretch_sound_rate" in params else (80,
                                                                          120)

        self.preprocessing_function = params[
            "preprocessing_function"] if "preprocessing_function" in params else None

    def augment_sound(self, x):
        if self.augment:
            x = self.random_augment_sound(x)
        if self.preprocessing_function is not None:
            x = self.preprocessing_function(x)
        if self.melsp:
            x = self.calculate_melsp(x)
        return x

    def random_augment_sound(self, x):
        # randomly add augmentation
        if np.random.choice((True, False)):
            # add white noise
            x = self.add_white_noise(
                x=x,
                rate=np.random.randint(self.white_noise_minmax[0],
                                       self.white_noise_minmax[1]) / 1000)
        _t = np.random.choice([0, 1, 2])
        if _t == 1:
            # shift sound
            x = self.shift_sound(
                x=x,
                rate=np.random.choice(
                    np.arange(self.shift_sound_minmax[0],
                              self.shift_sound_minmax[1])))
        elif _t == 2:
            # stretch sound
            x = self.stretch_sound(
                x=x,
                rate=np.random.choice(
                    np.arange(self.stretch_sound_minmax[0],
                              self.stretch_sound_minmax[1])) / 100)
        return x

    # change wave data to mel-stft
    def calculate_melsp(self, x):
        stft = np.abs(
            librosa.stft(
                x, n_fft=self.melsp_n_fft,
                hop_length=self.melsp_hop_length))**2
        log_stft = librosa.power_to_db(stft)
        melsp = librosa.feature.melspectrogram(
            S=log_stft, n_mels=self.melsp_n_mels)
        return melsp

    # data augmentation: add white noise
    def add_white_noise(self, x, rate=0.002):
        return x + rate * np.random.randn(len(x))

    # data augmentation: shift sound in timeframe
    def shift_sound(self, x, rate=2):
        return np.roll(x, int(len(x) // rate))

    # data augmentation: stretch sound
    def stretch_sound(self, x, rate=1.1):
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate)
        if len(x) > input_length:
            return x[:input_length]
        else:
            return np.pad(x, (0, max(0, input_length - len(x))), "constant")