import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from script import Score
import pandas as pd
from unittest.mock import patch
from io import StringIO

def check_sampled_piano_roll(score_path, truth_path):
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
  piece = Score(score_path)
  lyrics = piece.lyrics()
  assert(type(lyrics) == pd.DataFrame)
  assert(lyrics.shape == shape)
  assert(lyrics.iloc[0].at[lyrics.iloc[0].first_valid_index()] == first)
  assert(lyrics.iloc[-1].at[lyrics.iloc[-1].last_valid_index()] == last)

def check_harm_spine(score, control, filler='forward', output='array'):
  test = score.harm(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

def check_harm_keys(score, control, filler='forward', output='array'):
  test = score.harmKeys(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

def check_function_spine(score, control, filler='forward', output='array'):
  test = score.functions(filler=filler, output=output)
  if output == 'array':
    assert all(test == control)
  else:  # data are pandas series
    assert test.equals(control)

# check creation of Score objects from various types of symbolic notation files
def test_local_import():
  assert isinstance(Score('./test_files/monophonic1note.mid'), Score)
  assert isinstance(Score('./test_files/monophonic3notes.mid'), Score)
  assert isinstance(Score('./test_files/monophonic6notes.mid'), Score)
  assert isinstance(Score('./test_files/polyExample1note.mid'), Score)
  assert isinstance(Score('./test_files/polyExample2voices1note.mid'), Score)
  assert isinstance(Score('./test_files/polyExample3voices1note.mid'), Score)
  assert isinstance(Score('./test_files/polyphonic4voices1note.mid'), Score)
  assert isinstance(Score('./test_files/polyphonic4voices1note.mei'), Score)

def test_remote_import():
  assert isinstance(Score('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn'), Score)

def test_volpiano_import_1():
  piece = Score('1---gkjH7--klk-jkjh-ghg---jh---kl--k---jh--jk--hj---ghjh--hg---g--g--g--g7---hjh-ghjh---gf--gh--hkjklkjh--hjh-ghg---3---f--g77---3')
  assert isinstance(piece, Score)
  assert not piece.notes().empty
  
def test_volpiano_import_2():
  piece = Score('1--c--d---f--d---ed--c--d---f---g--h--j---hgf--g--h---')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_volpiano_import_3():
  piece = Score('./test_files/volpiano_example.txt')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_volpiano_import_4():
  piece = Score('./test_files/volpiano_explicit_example.txt')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_tinyNotation_import_1():
  piece = Score('4/4 c4 c# c c## cn c- c-- c_Lyric c1')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_tinyNotation_import_2():
  piece = Score('tinyNotation: 4/4 c4 trip{c8 d e} trip{f4 g a} b-1')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_tinyNotation_import_3():
  piece = Score('./test_files/tinyNotation_example.txt')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_tinyNotation_import_4():
  piece = Score('./test_files/tinyNotation_explicit_example.txt')
  assert isinstance(piece, Score)
  assert not piece.notes().empty

def test_lyrics():
  check_lyrics('https://raw.githubusercontent.com/alexandermorgan/AMPACT/main/test_files/busnoys.krn', (438, 4), 'In', 'cum.')

def test_spine_data():
  piece = Score('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K025/Stripped/M025_00_01a_a.krn')
  harm_series = pd.Series(['V', 'I', 'I', 'I', 'I', 'I', 'I', 'V', 'V', 'V', 'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'I',
      'V', 'V', 'V7b', 'V7b', 'I', 'I', 'I', 'iib', 'iib', 'iib', 'iib', 'V', 'V', 'I', 'I', 'I'],
    [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0,
      14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 19.0, 19.5, 20.0, 20.5, 21.0, 22.0, 23.0])
  check_harm_spine(piece, harm_series.values)
  check_harm_spine(piece, harm_series, output='series')
  check_harm_keys(piece, pd.Series(['D'], [0.0]), filler='drop', output='series')
  check_function_spine(piece, pd.Series(['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T',
      'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'P', 'P', 'P', 'P', 'D', 'D', 'T', 'T', 'T'], harm_series.index), output='series')

def test_show():
  '''\tMake sure the url generation works in a very simple case. Only check the first 100 characters
  because pyAMPACT adds the date of generation of the kern score in the footer metadata, so the full
  string is a little different every day.'''
  piece = Score('./test_files/showTest.krn')
  with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
    piece.show()
  expected_output = "https://verovio.humdrum.org/?t=ISEhQ09NOiBDb21wb3NlciBub3QgZm91bmQKISEhT1RMOiBUaXRsZSBub3QgZm91bmQKK"
  assert mock_stdout.getvalue()[:100] == expected_output

# def test_sampled():
#   doesn't work because music21 fills in missing rests in midi pieces
#   test = Score('./test_files/monophonic1note.mid').sampled()
#   control = pd.read_csv('./test_files/monophonic1note-pr.csv', header=None, index_col=False)
#   assert test.equals(control)

#   doesn't work because music21 fills in missing rests
#   test = Score('./test_files/monophonic3notes.mid').sampled()
#   control = pd.read_csv('./test_files/monophonic3notes-pr.csv', header=None, index_col=False)
#   assert test.equals(control)

#   doesn't work because music21 fills in missing rests
#   test = Score('./test_files/polyphonic4voices1note.mei').sampled()
#   control = pd.read_csv('./test_files/polyphonic4voices1note-pr.csv', header=None, index_col=False)
#   assert test.equals(control)


# check_sampled_piano_roll('./test_files/polyphonic4voices1note.mid', './test_files/polyphonic4voices1note-pr.csv')  # problematic because of terminal rests
