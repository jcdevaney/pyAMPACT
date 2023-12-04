# COMPLETE NEEDS TESTING
import numpy as np
import matplotlib.pyplot as plt

from midi2nmat import midi2nmat


def alignment_visualizer(trace, mid, spec, fig=1):
    if fig is None:
        fig = 1

    # # COMMENT OUT FOR > 1 NOTES
    # if trace[1, -1] == 0:
    #     trace = trace[:, :-1]

    # if trace[1, -2] == 0:
    #     trace[1, -2] = trace[1, -3]

    # hop size between frames
    stft_hop = 0.0228  # Adjusted from 0.025

    # Read MIDI file
    # note, vel, start time, end time, duration, note is processed
    nmat = midi2nmat(mid)

    # ADJUST CONTRAST...
    # Plot spectrogram of the audio file
    fig = plt.figure(fig)
    plt.imshow(20 * np.log10(spec), aspect='auto', origin='lower', cmap='gray')
    plt.title('Spectrogram with Aligned MIDI Notes Overlaid')
    plt.xlabel('Time (.05s)')
    plt.ylabel('Midinote')
    plt.clim([plt.gci().get_clim()[0], plt.gci().get_clim()[1] - 50])
    plt.colorbar()
    # plt.show() # Uncomment to show

    # Zoom in on fundamental frequencies
    notes = nmat[:, 0]  # Note
    notes = (2 ** ((notes - 105) / 12)) * 440
    notes = np.append(notes, notes[-1])
    nlim = len(notes)

    plot_fine_align(trace[0, :], trace[1, :],
                    notes[:nlim], stft_hop)  # Original

    # if trace.shape[0] >= 3: # Original

    #     if len(trace) >= 3:
    #         notenums = trace[2, 1:]
    #         # notenums = trace[2][1:]
    #     else:
    #         nlim = len(notes)
    #         notenums = np.concatenate([np.repeat(range(1, nlim + 1), 4), [nlim]])


def plot_fine_align(stateType, occupancy, notes, stftHop):
    """
    Plot the HMM alignment based on the output of YIN.

    Parameters:
        stateType: List of states in the HMM.
        occupancy: List indicating the time (in seconds) at which the states in stateType end.
        notes: List of MIDI note numbers that are played.
        stftHop: The hop size between frames in the spectrogram.
    """

    # Define styles for different states
    styles = [
        # {'color': 'red', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'none', 'marker': '+', 'linestyle': '-',
            'linewidth': 2},  # RED LINE RUNNING THROUGH?
        {'color': 'green', 'marker': '+', 'linestyle': '-', 'linewidth': 2},
        {'color': 'blue', 'marker': '+', 'linestyle': '-', 'linewidth': 2}]

    # Calculate segment boundaries
    cs = np.array(occupancy) / stftHop
    segments = np.vstack((cs[:-1], cs[1:])).T

    # Create the plot
    stateNote = (np.maximum(1, np.cumsum(stateType == 3) + 1)) - 1
    for i in range(segments.shape[0]):
        style = styles[int(stateType[i + 1]) - 1]
        x = segments[i, :]
        y = np.tile(notes[stateNote[i]], (2, 1))
        plt.plot(x, y, color=style['color'], marker=style['marker'],
                 linestyle=style['linestyle'], linewidth=style['linewidth'])

    plt.show()
