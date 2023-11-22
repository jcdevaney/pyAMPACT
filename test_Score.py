import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from script import Score
import pandas as pd

def check_sampled_piano_roll(score_path, truth_path):
  print(f'Running check_sampled_piano_roll on {score_path} and {truth_path} ...')
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

def check_lyrics(score_path, shape, first, last):
  print(f'Running check_lyrics on {score_path} ...')
  piece = Score(score_path)
  lyrics = piece.lyrics()
  assert(type(lyrics) == pd.DataFrame)
  assert(lyrics.shape == shape)
  assert(lyrics.iloc[0].at[lyrics.iloc[0].first_valid_index()] == first)
  assert(lyrics.iloc[-1].at[lyrics.iloc[-1].last_valid_index()] == last)

def check_harm_spine(score_path, control=pd.Series(), filler='forward', output='array'):
  print(f'Running check_harm_spine on {score_path} ...')
  piece = Score(score_path)
  test = piece.harmonies(filler=filler, output=output)
  assert test.equals(control)

def check_harm_keys(score_path, control=pd.Series(), filler='forward', output='array'):
  print(f'Running check_harm_keys on {score_path} ...')
  piece = Score(score_path)
  test = piece.harmKeys(filler=filler, output=output)
  assert test.equals(control)

def check_function_spine(score_path, control=pd.Series(), filler='forward', output='array'):
  print(f'Running check_function_spine on {score_path} ...')
  piece = Score(score_path)
  test = piece.functions(filler=filler, output=output)
  assert test.equals(control)

# check creation of Score objects from various types of symbolic notation files
def test_import():
  Score('./test_files/monophonic1note.mid')
  Score('./test_files/monophonic3notes.mid')
  Score('./test_files/monophonic6notes.mid')
  Score('./test_files/polyExample1note.mid')
  Score('./test_files/polyExample2voices1note.mid')
  Score('./test_files/polyExample3voices1note.mid')
  Score('./test_files/polyphonic4voices1note.mid')
  Score('./test_files/polyphonic4voices1note.mei')

def test_all():
  piece = Score('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn')
  # check_lyrics('./test_files/busnoys.krn', (438, 4), 'In', 'cum.')
  check_harm_spine('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn',
    pd.Series(['V', 'I', 'I', 'I', 'I', 'I', 'I', 'V', 'V', 'V', 'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'I', 'V', 'V', 'V7b',
                'V7b', 'I', 'I', 'I', 'iib', 'iib', 'iib', 'iib', 'V', 'V', 'I', 'I', 'I'],
              [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0]), output='series')
  check_harm_keys('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn',
    pd.Series(['D'], [0.0]), filler='drop', output='series')
  check_function_spine('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn',
    pd.Series(['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
                'T', 'T', 'T', 'P', 'P', 'P', 'P', 'D', 'D', 'T', 'T', 'T'],
              [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0]), output='series')
  

# check_sampled_piano_roll('./test_files/polyphonic4voices1note.mid', './test_files/polyphonic4voices1note-pr.csv')  # problematic because of terminal rests
# check_sampled_piano_roll('./test_files/polyphonic4voices1note.mei', './test_files/polyphonic4voices1note-pr.csv')  # problematic because of mei encoding
# check_sampled_piano_roll('./test_files/monophonic1note.mid', './test_files/monophonic1note-pr.csv')  # doesn't work because groundTruth is missing a row of zeros at beginning
# check_sampled_piano_roll('./test_files/monophonic3notes.mid', './test_files/monophonic3notes-pr.csv')
# check_sampled_piano_roll('./test_files/B063_00_01a_a.mei', './test_files/polyphonic4voices1note-pr.csv')