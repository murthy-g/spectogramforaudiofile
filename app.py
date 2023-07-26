
import librosa

audio, sampling_rate = librosa.load('./audio.wav', sr = None, mono = True, offset = 0.0, duration = None)

import matplotlib.pyplot as plt
plt.figure().set_figwidth(12)
librosa.display.waveshow(audio,sr = sampling_rate)



## Frequency Spectrum
import numpy as np

input_data = audio[:4096]

window = np.hanning(len(input_data))
# print(window)

dft_input = input_data * window

figure = plt.figure(figsize = (15,5))
plt.subplot(131)
plt.plot(input_data)
plt.title('input')
plt.subplot(132)
plt.plot(window)
plt.title('hanning window')
plt.subplot(133)
plt.plot(dft_input)
plt.title('dft_input')
plt.show()
# similar plot is generated for every instance

# calculate the dft - discrete fourier transform
dft = np.fft.rfft(dft_input)
plt.plot(dft)

# amplitude 
amplitude = np.abs(dft)
# convert it into dB
amplitude_dB = librosa.amplitude_to_db(amplitude,ref = np.max)


# sometimes people want to use power spectrum -> A**2

print(len(amplitude))
print(len(dft_input))
print(len(dft))

# frequency
frequency = librosa.fft_frequencies(n_fft=len(input_data),sr = sampling_rate)
plt.figure().set_figwidth(12)
plt.plot(frequency, amplitude_dB)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.xscale("log")

plt.show()
spectogram = librosa.stft(audio)
spectogram_to_dB = librosa.amplitude_to_db(np.abs(spectogram),ref = np.max)
plt.figure().set_figwidth(12)
librosa.display.specshow(spectogram_to_dB, x_axis="time", y_axis="hz")
plt.colorbar()

MelSpectogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=128, fmax=8000)
MelSpectogram_dB = librosa.power_to_db(MelSpectogram, ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(MelSpectogram_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
plt.show()