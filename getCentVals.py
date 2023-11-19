# COMPLETE NEEDS TESTING
def get_cent_vals(times, yinres, sr):
    cents = []

    # Index into f0 estimates in YIN structure with onset and offset times
    for i in range(len(times)):
        onset_sample = int(times[i] / 32 * sr)

        # Extract the corresponding portion of f0 and convert to cents
        if onset_sample < len(yinres['f0']):
            f0_segment = yinres['f0'][onset_sample] * 1200
            cents.append(f0_segment)

    return cents


# # COMPLETE NEEDS TESTING

# def get_cent_vals(times, yinres):
#     cents = []

#     # Index into f0 estimates in YIN structure with onset and offset times
#     for i in range(len(times['ons'])):
#         onset_sample = int(times['ons'][i] / 32 * yinres['sr'])
#         offset_sample = int(times['offs'][i] / 32 * yinres['sr'])

#         # Extract the corresponding portion of f0 and convert to cents
#         f0_segment = yinres['f0'][onset_sample:offset_sample] * 1200
#         cents.append(f0_segment)

#     return cents
