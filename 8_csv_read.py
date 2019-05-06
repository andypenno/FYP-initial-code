import csv
import matplotlib.pyplot as plt
import numpy as np

plt.ioff() ## Turns interactive plotting off (allows plt figures to be created, saved and then closed without displaying)


recording_file = "charlotte2.csv"
x = []
y= []

with open(recording_file) as csvfile:
    read_csv = csv.reader(csvfile, delimiter='\t')
    i = 0
    for row in read_csv:
        i += 1
        if i >= 7:
            x.append(float(row[0]))
            y.append(float(row[1]))

#for i in range(0, len(x)):
#    x[i] = 0.1 - x[i]
#    y[i] = 0.1 - y[i]

fig = plt.figure(figsize=(100, 10)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

plt.subplot(2, 1, 1)
plt.plot(x, 'k')

plt.subplot(2, 1, 2)
plt.plot(y, 'k')


csv_plot_name = "plot_ppg.png"
fig.savefig(csv_plot_name) ## Saves the output figure to a .png image file
print("\nPPG data figure saved to: " + csv_plot_name)
plt.close(fig)
      
fft_x = abs(np.fft.fft(x))
fft_y = abs(np.fft.fft(y))


fig = plt.figure(figsize=(10, 10)) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

plt.subplot(2, 1, 1)
plt.plot(fft_x, 'k')

plt.subplot(2, 1, 2)
plt.plot(fft_y, 'k')
    
csv_fft_plot_name = "plot_ppg_fft.png"
fig.savefig(csv_fft_plot_name) ## Saves the output figure to a .png image file
print("\nPPG FFT data figure saved to: " + csv_fft_plot_name)
plt.close(fig)
























