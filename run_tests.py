import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from script import Score

def test_sampled_piano_roll(score_path, truth_path):
  print(f'Running test_sampled_piano_roll on {score_path} and {truth_path} ...')
  piece = Score(score_path)
  pianoRoll = piece.piano_roll()
  sampled = piece.sampled()
  samp = sampled.copy()
  sampled.columns = range(sampled.columns.size)  # reset column names to be enumerated
  groundTruth = pd.read_csv(truth_path, header=None)
  groundTruth.index = list(range(1, 128))
  slice1 = sampled.loc[1:127, :]
  slice2 = groundTruth.loc[1:127, :]
  assert(slice1.equals(slice2))

def test_lyrics(score_path, shape, first, last):
  print(f'Running test_lyrics on {score_path} ...')
  piece = Score(score_path)
  lyrics = piece.lyrics()
  assert(type(lyrics) == pd.DataFrame)
  assert(lyrics.shape == shape)
  assert(lyrics.iloc[0].at[lyrics.iloc[0].first_valid_index()] == first)
  assert(lyrics.iloc[-1].at[lyrics.iloc[-1].last_valid_index()] == last)

def test_harm_spine(score_path, controlHarmonies, controlKeys):
  print(f'Running test_harm_spine on {score_path} ...')
  piece = Score(score_path)
  test1 = piece.harmonies().reindex_like(controlHarmonies)
  test2 = piece.harmKeys().reindex_like(controlKeys)
  assert test1.equals(controlHarmonies)
  assert test2.equals(controlKeys)

def test_function_spine(score_path, controlFunctions):
  print(f'Running test_harm_spine on {score_path} ...')
  piece = Score(score_path)
  test = piece.functions().reindex_like(controlFunctions)
  assert test.equals(controlFunctions)

# check creation of Score objects from various types of symbolic notation files
piece = Score('./test_files/M025_00_01a_a-repeated.krn')
Score('./test_files/monophonic1note.mid')
Score('./test_files/monophonic3notes.mid')
Score('./test_files/monophonic6notes.mid')
Score('./test_files/polyExample1note.mid')
Score('./test_files/polyExample2voices1note.mid')
Score('./test_files/polyExample3voices1note.mid')
Score('./test_files/polyphonic4voices1note.mid')
Score('./test_files/polyphonic4voices1note.mei')
Score('./test_files/busnoys.krn')
Score('./test_files/B063_00_01a_a.mei')  # harmonic analysis is encoded and analyzed as lyrics
test_lyrics('./test_files/busnoys.krn', (438, 4), 'In', 'cum.')
test_harm_spine('./test_files/M025_00_01a_a-repeated.krn',
  pd.Series(['V', 'I', 'I', 'I', 'I', 'I', 'V', 'V', 'V', 'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'I', 'V', 'V', 'V7b',
              'V7b', 'I', 'I', 'I', 'iib', 'iib', 'iib', 'iib', 'V', 'V', 'I', 'I', 'I'],
            [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
              14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0]),
  pd.Series(['*D:'], [0.0]))
test_function_spine('./test_files/M025_00_01a_a-repeated.krn',
  pd.Series(['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
              'T', 'T', 'T', 'P', 'P', 'P', 'P', 'D', 'D', 'T', 'T', 'T'],
            [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
              14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0]))

print('\n\n\t\t**** All tests passed ✅ ****\n\n')
  

# test_sampled_piano_roll('./test_files/polyphonic4voices1note.mid', './test_files/polyphonic4voices1note-pr.csv')  # problematic because of terminal rests
# test_sampled_piano_roll('./test_files/polyphonic4voices1note.mei', './test_files/polyphonic4voices1note-pr.csv')  # problematic because of mei encoding
# test_sampled_piano_roll('./test_files/monophonic1note.mid', './test_files/monophonic1note-pr.csv')  # doesn't work because groundTruth is missing a row of zeros at beginning
# test_sampled_piano_roll('./test_files/monophonic3notes.mid', './test_files/monophonic3notes-pr.csv')
# test_sampled_piano_roll('./test_files/B063_00_01a_a.mei', './test_files/polyphonic4voices1note-pr.csv')