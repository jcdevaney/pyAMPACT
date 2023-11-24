import pandas as pd
import numpy as np
import music21 as m21
import math
import ast
import pdb
import json
import requests
import os
import tempfile
import re
m21.environment.set('autoDownload', 'allow')

function_pattern = re.compile('[^TtPpDd]')
imported_scores = {}
_duration2Kern = {  # keys get rounded to 5 decimal places
  56: '000..',
  48: '000.',
  32: '000',
  28: '00..',
  24: '00.',
  16: '00',
  14: '0..',
  12: '0.',
  8: '0',
  7: '1..',
  6: '1.',
  4: '1',
  3.5: '2..',
  3: '2.',
  2.66666: '3%2',
  2: '2',
  1.75: '4..',
  1.5: '4.',
  1.33333: '3',
  1: '4',
  .875: '8..',
  .75: '8.',
  .66667: '6',
  .5: '8',
  .4375:  '16..',
  .375:   '16.',
  .33333: '12',
  .25:    '16',
  .21875: '32..',
  .1875:  '32.',
  .16667: '24',
  .125:   '32',
  .10938: '64..',
  .09375: '64.',
  .08333: '48',
  .0625:  '64',
  .05469: '128..',
  .04688: '128.',
  .04167: '96',
  .03125: '128',
  .02734: '256..',
  .02344: '256.',
  .02083: '192',
  .01563: '256',
  0:      ''
}

class Score:
  '''\tImport score via music21 and expose AMPACT's analysis utilities which are
  generally formatted as Pandas DataFrames.'''
  def __init__(self, score_path):
    self._analyses = {}
    self.path = score_path
    self._tempFile = ''
    self.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    self.fileExtension = score_path.rsplit('.', 1)[1]
    if score_path.startswith('http') and self.fileExtension == 'krn':
      fd, tmp_path = tempfile.mkstemp()
      try:
        with os.fdopen(fd, 'w') as tmp:
          response = requests.get(self.path)
          tmp.write(response.text)
          tmp.seek(0)
          self._assignM21Attributes(tmp_path)
          self._import_function_harm_spines(tmp_path)
      finally:
        os.remove(tmp_path)
    else:  # file is not an online kern file (can be either or neither but not both)
      self._assignM21Attributes()
      self._import_function_harm_spines()
    self.public = '\n'.join([f'{prop.ljust(15)}{type(getattr(self, prop))}' for prop in dir(self) if not prop.startswith('_')])
  
  def _assignM21Attributes(self, path=''):
    '''\tReturn a music21 score. This method is used internally for memoization purposes.'''
    if self.path not in imported_scores:
      if path:
        imported_scores[self.path] = m21.converter.parse(path, format='humdrum')
      else:
        imported_scores[self.path] = m21.converter.parse(self.path)
    self.score = imported_scores[self.path]
    self.metadata = {'Title': self.score.metadata.title, 'Composer': self.score.metadata.composer}
    self._partStreams = self.score.getElementsByClass(m21.stream.Part)
    self._semiFlatParts = []
    self.partNames = []
    for i, part in enumerate(self._partStreams):
      part.makeMeasures(inPlace=True)
      self._semiFlatParts.append(part.flatten())
      name = part.partName if (part.partName and part.partName not in self.partNames) else 'Part_' + str(i + 1)
      self.partNames.append(name)

  def _partList(self):
    '''\tReturn a list of series of the note, rest, and chord objects in a each part.'''
    if '_partList' not in self._analyses:
      parts = []
      isUnique = True
      for i, flat_part in enumerate(self._semiFlatParts):
        ser = pd.Series([nrc for nrc in flat_part.getElementsByClass(['Note', 'Rest', 'Chord'])], name=self.partNames[i])
        ser.index = ser.apply(lambda nrc: nrc.offset).astype(float).round(5)
        # ser = ser[~ser.index.duplicated(keep='last')]
        if not ser.index.is_unique:
          isUnique = False
        parts.append(ser)
      if not isUnique:
        for part in parts:
          tieBreakers = []
          nexts = part.index.to_series().shift(-1)
          for i in range(-1, -1 - len(part.index), -1):
            if part.index[i] == nexts.iat[i]:
              tieBreakers.append(tieBreakers[-1] - 1)
            else:
              tieBreakers.append(0)
          tieBreakers.reverse()
          part.index = pd.MultiIndex.from_arrays((part.index, tieBreakers))
      self._analyses['_partList'] = parts
    return self._analyses['_partList']

  def _parts(self, multi_index=False):
    '''\tReturn a df of the note, rest, and chord objects in the score. The difference between
    parts and divisi is that parts can have chords whereas divisi cannot. If there are chords
    in the _parts df, the divisi df will include all these notes by adding additional columns.'''
    key = ('_parts', multi_index)
    if key not in self._analyses:
      df = pd.concat(self._partList(), axis=1, sort=True)
      if not multi_index and isinstance(df.index, pd.MultiIndex):
        df.index = df.index.droplevel(1)
      self._analyses[key] = df
    return self._analyses[key]

  def _divisi(self, multi_index=False):
    '''\tReturn a df of the note and rest objects in the score without chords. The difference between
    parts and divisi is that parts can have chords whereas divisi cannot. If there are chords
    in the _parts df, the divisi df will include all these notes by adding additional columns.'''
    key = ('_divisi', multi_index)
    if key not in self._analyses:
      divisi = []
      for i, part in enumerate(self._partList()):
        div = part.apply(lambda nrc: nrc.notes if nrc.isChord else (nrc,)).apply(pd.Series)
        if len(div.columns) > 1:
          div.columns = [':'.join((self.partNames[i], str(j))) for j in range(1, len(div.columns) + 1)]
          div.ffill(axis=1, inplace=True)
        else:
          div.columns = [self.partNames[i]]
        divisi.append(div)
      df = pd.concat(divisi, axis=1, sort=True)
      if not multi_index and isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(1)
      self._analyses[key] = df
    return self._analyses[key]

  def _import_function_harm_spines(self, path=''):
    if self.fileExtension == 'krn' or path:
      humFile = m21.humdrum.spineParser.HumdrumFile(path or self.path)
      humFile.parseFilename()
      for spine in humFile.spineCollection:
        if spine.spineType in ('harm', 'function', 'cdata'):
          start = False
          vals, valPositions = [], []
          if spine.spineType == 'harm':
            keyVals, keyPositions = [], []
          for i, event in enumerate(spine.eventList):
            contents = event.contents
            if contents.endswith(':') and contents.startswith('*'):
              start = True
              # there usually won't be any m21 objects at the same position as the key events,
              # so use the position from the next item in eventList if there is a next item.
              if spine.spineType == 'harm' and i + 1 < len(spine.eventList):
                keyVals.append(contents[1:-1])     # [1:-1] to remove the * and : characters
                keyPositions.append(spine.eventList[i+1].position)
              continue
            elif not start or '!' in contents or '=' in  contents or '*' in contents:
              continue
            else:
              if spine.spineType == 'function':
                func = function_pattern.sub('', contents)
                if len(func):
                  vals.append(func)
                else:
                  continue
              else:
                vals.append(contents)
              valPositions.append(event.position)

          df1 = self._priority()
          name = spine.spineType.title()
          if name == 'Cdata':
            df2 = pd.DataFrame([ast.literal_eval(val) for val in vals], index=valPositions)
          else:
            df2 = pd.DataFrame({name: vals}, index=valPositions)
          joined = df1.join(df2, on='Priority')
          if name != 'Cdata':   # get all the columns from the third to the end. Usually just 1 col except for cdata
            res = joined.iloc[:, 2].copy()
          else:
            res = joined.iloc[:, 2:].copy()
          res.index = joined['Offset']
          res.index.name = ''
          self._analyses[spine.spineType] = res
          if spine.spineType == 'harm' and len(keyVals):
            keyName = 'harmKeys'
            # key records are usually not found at a kern line with notes so take the next valid one
            keyPositions = [df1.iat[np.where(df1.Priority >= kp)[0][0], 0] for kp in keyPositions]
            df3 = pd.DataFrame({keyName: keyVals}, index=keyPositions)
            joined = df1.join(df3, on='Priority')
            ser = joined.iloc[:, 2].copy()
            ser.index = joined['Offset']
            ser.index.name = ''
            self._analyses[keyName] = ser

    if 'function' not in self._analyses:
      self._analyses['function'] = pd.Series()
    if 'harm' not in self._analyses:
      self._analyses['harm'] = pd.Series()
    if 'harmKeys' not in self._analyses:
      self._analyses['harmKeys'] = pd.Series()
    if 'cdata' not in self._analyses:
      self._analyses['cdata'] = pd.DataFrame()

  def lyrics(self):
    if 'lyrics' not in self._analyses:
      self._analyses['lyrics'] = self._divisi().applymap(lambda cell: cell.lyric or np.nan, na_action='ignore').dropna(how='all')
    return self._analyses['lyrics']

  def _clefHelper(self, clef):
    '''\tParse a music21 clef object into the corresponding humdrum syntax token.'''
    octaveChange = ''
    if clef.octaveChange > 0:
      octaveChange = '^' * clef.octaveChange
    elif clef.octaveChange < 0:
      octaveChange = 'v' * abs(clef.octaveChange)
    return f'*clef{clef.sign}{octaveChange}{clef.line}'

  def _clefs(self):
    if 'clefs' not in self._analyses:
      parts = []
      isUnique = True
      for i, flat_part in enumerate(self._semiFlatParts):
        ser = pd.Series(flat_part.getElementsByClass(['Clef']), name=self.partNames[i])
        ser.index = ser.apply(lambda nrc: nrc.offset).astype(float).round(5)
        # ser = ser[~ser.index.duplicated(keep='last')]
        if not ser.index.is_unique:
          isUnique = False
        parts.append(ser)
      if not isUnique:
        for part in parts:
          tieBreakers = []
          nexts = part.index.to_series().shift(-1)
          for i in range(-1, -1 - len(part.index), -1):
            if part.index[i] == nexts.iat[i]:
              tieBreakers.append(tieBreakers[-1] - 1)
            else:
              tieBreakers.append(0)
          tieBreakers.reverse()
          part.index = pd.MultiIndex.from_arrays((part.index, tieBreakers))
      clefs = pd.concat(parts, axis=1)
      if isinstance(clefs.index, pd.MultiIndex):
        clefs = clefs.droplevel(1)
      self._analyses['clefs'] = clefs.applymap(self._clefHelper, na_action='ignore')
    return self._analyses['clefs']

  def dynamics(self):
    if 'dynamics' not in self._analyses:
      dyns = [pd.Series({obj.offset: obj.value for obj in sf.getElementsByClass('Dynamic')}) for sf in self._semiFlatParts]
      dyns = pd.concat(dyns, axis=1)
      dyns.columns = self.partNames
      dyns.dropna(how='all', axis=1, inplace=True)
      self._analyses['dynamics'] = dyns
    return self._analyses['dynamics']

  def _priority(self):
    '''\tFor .krn files, get the line numbers of the events in the piece, which music21
    often calls "priority". For other encoding formats return an empty dataframe.'''
    if '_priority' not in self._analyses:
      if self.fileExtension != 'krn':
        priority = pd.DataFrame()
      else:
        priority = self._parts().applymap(lambda cell: cell.priority, na_action='ignore').ffill(axis=1).iloc[:, -1].astype('Int16')
        priority = pd.DataFrame({'Priority': priority.values, 'Offset': priority.index})
      self._analyses['_priority'] = priority
    return self._analyses['_priority']

  def _snapTo(self, data, snap_to=None, filler='forward', output='array'):
    '''\tTakes a `harmonies`, `harmKeys`, `functions`, or `cdata` as `data` and the
    `snap_to` and `filler` parameters as described in the former three's doc strings.
    The passed data is returned in the shape of the snap_to dataframe's columns, and any
    filling operations are applied. The output will be in the form of a 1D numpy array
    unless `output` is changed, in which case a series will be returned for harmonies,
    harmKeys, and functions data, and a dataframe for cdata data.'''
    if snap_to is not None:
      data = data.reindex(snap_to.columns)
    if filler != '.':
      data.replace('.', np.nan, inplace=True)
    if isinstance(filler, str):
      filler = filler.lower()
      if filler == 'forward':
        data.ffill(inplace=True)
      else:
        if filler in ('nan', 'drop'):
          data.fillna(np.nan, inplace=True)
        else:
          data.fillna(filler, inplace=True)
    if filler == 'drop':
      data.dropna(inplace=True)
    if output == 'array':
      return data.values
    else:
      return data

  def harmKeys(self, snap_to=None, filler='forward', output='array'):
    '''\tGet the key analysis from the **harm spine if this piece is a kern file and has a
    **harm spine. The default is for the results to be returned as a 1-d array, but you can
    set `output='series'` for a pandas series instead. If you want to align these results
    so that they match the columnar (time) axis of the pianoRoll, sampled, or mask results,
    you can pass the pianoRoll or mask that you want to align to as the `snap_to` parameter.

    The `sampled` and `mask` will almost always have more observations than the `harmKeys`
    results, so you may want to fill in these new empty slots somehow. The kern format uses
    '.' as a filler token so you can pass this as the `filler` parameter to fill all the new
    empty slots with this as well. If you choose some other value, say `filler='_'`, then in
    addition to filling in the empty slots with underscores, this will also replace the kern
    '.' observations with '_'. If you want to fill them in with NaN's as pandas usually does,
    you can pass `filler='nan'` as a convenience. If you want to "forward fill" these
    results, you can pass `filler='forward'` (default). This will propagate the last
    non-period ('.') observation until a new one is found. Finally, you can pass filler='drop'
    to drop all empty observations (both NaNs and humdrum periods).

    Usage assuming you have a Score object named `piece` in memory:
    # get the key data as a forward-filled array. No need to specify filler='forward' because it's the default
    keys = piece.harmKeys()

    # get the harmonies in the shape of the mask columns
    mask = piece.mask()
    keys = piece.harmKeys(snap_to=mask)

    # get the harmonies in the shape of the mask columns and replace kern's '.' tokens with NaNs
    mask = piece.mask()
    keys = piece.harmKeys(snap_to=mask, filler='nan')
    '''
    return self._snapTo(self._analyses['harmKeys'].copy(), snap_to, filler, output)

  def harmonies(self, snap_to=None, filler='forward', output='array'):
    '''\tGet the harmonic analysis from the **harm spine if this piece is a kern file and has a
    **harm spine. The default is for the results to be returned as a 1-d array, but you can
    set `output='series'` for a pandas series instead. If you want to align these results
    so that they match the columnar (time) axis of the pianoRoll, sampled, or mask results,
    you can pass the pianoRoll or mask that you want to align to as the `snap_to` parameter.

    The `sampled` and `mask` will almost always have more observations than the `harmonies`
    results, so you may want to fill in these new empty slots somehow. The kern format uses
    '.' as a filler token so you can pass this as the `filler` parameter to fill all the new
    empty slots with this as well. If you choose some other value, say `filler='_'`, then in
    addition to filling in the empty slots with underscores, this will also replace the kern
    '.' observations with '_'. If you want to fill them in with NaN's as pandas usually does,
    you can pass `filler='nan'` as a convenience. If you want to "forward fill" these
    results, you can pass `filler='forward'` (default). This will propagate the last
    non-period ('.') observation until a new one is found. Finally, you can pass filler='drop'
    to drop all empty observations (both NaNs and humdrum periods).

    Usage assuming you have a Score object named `piece` in memory:
    # get the harm data as a forward-filled array. No need to specify filler='forward' because it's the default
    harmonies = piece.harmonies()

    # get the harmonies in the shape of the mask columns
    mask = piece.mask()
    harmonies = piece.harmonies(snap_to=mask)

    # get the harmonies in the shape of the mask columns and replace kern's '.' tokens with NaNs
    mask = piece.mask()
    harmonies = piece.harmonies(snap_to=mask, filler='nan')
    '''
    return self._snapTo(self._analyses['harm'].copy(), snap_to, filler, output)

  def functions(self, snap_to=None, filler='forward', output='array'):
    '''\tGet the functional analysis from the **function spine if this piece is a kern file and
    has a **function spine. The default is for the results to be returned as a 1-d array, but
    you can set `output='series'` for a pandas series instead. If you want to align these results
    so that they match the columnar (time) axis of the pianoRoll, sampled, or mask results,
    you can pass the pianoRoll or mask that you want to align to as the `snap_to` parameter.

    The `sampled` and `mask` will almost always have more observations than the `harmKeys`
    results, so you may want to fill in these new empty slots somehow. The kern format uses
    '.' as a filler token so you can pass this as the `filler` parameter to fill all the new
    empty slots with this as well. If you choose some other value, say `filler='_'`, then in
    addition to filling in the empty slots with underscores, this will also replace the kern
    '.' observations with '_'. If you want to fill them in with NaN's as pandas usually does,
    you can pass `filler='nan'` as a convenience. If you want to "forward fill" these
    results, you can pass `filler='forward'` (default). This will propagate the last
    non-period ('.') observation until a new one is found. Finally, you can pass filler='drop'
    to drop all empty observations (both NaNs and humdrum periods).

    Usage assuming you have a Score object named `piece` in memory:
    # get the functional analysis as a forward-filled array. No need to specify filler='forward' because it's the default
    functions = piece.functions()

    # get the functions in the shape of the mask columns
    mask = piece.mask()
    functions = piece.functions(snap_to=mask)

    # get the functions in the shape of the mask columns and replace kern's '.' tokens with NaNs
    mask = piece.mask()
    functions = piece.functions(snap_to=mask, filler='nan')
    '''
    return self._snapTo(self._analyses['function'].copy(), snap_to, filler, output)

  def cdata(self, snap_to=None, filler='forward', output='dataframe'):
    '''\tGet the key analysis from the **cdata spine if this piece is a kern file and has a
    **cdata spine. The default is for the results to be returned as a pandas.DataFrame. 
    If you want to align these results so that they match the columnar (time) axis of the
    pianoRoll, sampled, or mask results, you can pass the pianoRoll or mask that you want to
    align to as the `snap_to` parameter.

    The `sampled` and `mask` will almost always have more observations than the `cdata`
    results, so you may want to fill in these new empty slots somehow. The kern format uses
    '.' as a filler token so you can pass this as the `filler` parameter to fill all the new
    empty slots with this as well. If you choose some other value, say `filler='_'`, then in
    addition to filling in the empty slots with underscores, this will also replace the kern
    '.' observations with '_'. If you want to fill them in with NaN's as pandas usually does,
    you can pass `filler='nan'` as a convenience. If you want to "forward fill" these
    results, you can pass `filler='forward'` (default). This will propagate the last
    non-period ('.') observation until a new one is found.

    Usage assuming you have a Score object named `piece` in memory:
    # get the cdata as a forward-filled dataframe. No need to specify filler='forward' because it's the default
    cdata = piece.cdata()

    # get the cdata in the shape of the mask columns
    mask = piece.mask()
    cdata = piece.cdata(snap_to=mask)

    # get the cdata in the shape of the mask columns and replace kern's '.' tokens with NaNs
    mask = piece.mask()
    cdata = piece.cdata(snap_to=mask, filler='nan')
    '''
    return self._snapTo(self._analyses['cdata'].copy(), snap_to, filler, output)

  def _remove_tied(self, noteOrRest):
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
      return np.nan
    return noteOrRest

  def _m21ObjectsNoTies(self):
    if '_m21ObjectsNoTies' not in self._analyses:
      self._analyses['_m21ObjectsNoTies'] = self._divisi(multi_index=True).applymap(self._remove_tied).dropna(how='all')
    return self._analyses['_m21ObjectsNoTies']

  def _measures(self, divisi=True):
    '''\tReturn df of the measure starting points.'''
    key = ('_measure', divisi)
    if key not in self._analyses:
      partMeasures = [pd.Series({m.offset: m.measureNumber for m in part.makeMeasures()}, dtype='Int16')
                      for i, part in enumerate(self._semiFlatParts)]
      df = pd.concat(partMeasures, axis=1)
      df.columns = self.partNames
      self._analyses[key] = df
    return self._analyses[key]

  def _barlines(self):
    '''\tReturn df of barlines specifying which barline type. Double barline, for
    example, can help detect section divisions, and the final barline can help
    process the `highestTime` similar to music21.'''
    if "_barlines" not in self._analyses:
      partBarlines = [pd.Series({m.offset: m.measureNumber for m in part.getElementsByClass(['Barline'])})
                      for i, part in enumerate(self._semiFlatParts)]
      df = pd.concat(partBarlines, axis=1)
      df.columns = self.partNames
      self._analyses["_barlines"] = df
    return self._analyses["_barlines"]

  def _keySignatures(self, kern=True):
    if '_keySignatures' not in self._analyses:
      kSigs = []
      for i, part in enumerate(self._semiFlatParts):
        kSigs.append(pd.Series({ky.offset: ky for ky in part.getElementsByClass(['Key'])}, name=self.partNames[i]))          
      df = pd.concat(kSigs, axis=1).sort_index(kind='mergesort')
      if kern:
        df = '*k[' + df.applymap(lambda ky: ''.join([_note.name for _note in ky.alteredPitches]).lower(), na_action='ignore') + ']'
      self._analyses['_keySignatures'] = df
    return self._analyses['_keySignatures']

  def _timeSignatures(self):
    if '_timeSignatures' not in self._analyses:
      tsigs = []
      for i, part in enumerate(self._semiFlatParts):
        tsigs.append(pd.Series({ts.offset: ts.ratioString for ts in part.getTimeSignatures()}, name=self.partNames[i]))
      df = pd.concat(tsigs, axis=1).sort_index(kind='mergesort')
      self._analyses['_timeSignatures'] = df
    return self._analyses['_timeSignatures']

  def durations(self, multi_index=False, df=None):
    '''\tReturn dataframe of durations of note and rest objects in piece.'''
    if df is None:
      key = ('durations', multi_index)
      if key not in self._analyses:
        m21objs = self._m21ObjectsNoTies()
        res = m21objs.applymap(lambda nrc: nrc.quarterLength, na_action='ignore').astype(float).round(5)
        if not multi_index and isinstance(res.index, pd.MultiIndex):
          res = res.droplevel(1)
        self._analyses[key] = res
      return self._analyses[key]
    else:   # df is not None so calculate diff between cell offsets per column in passed df, skip memoization
      sers = []
      for col in range(len(df.columns)):
        part = df.iloc[:, col].dropna()
        ndx = part.index.get_level_values(0)
        if len(part) > 1:
          vals = (ndx[1:] - ndx[:-1]).to_list()
        else:
          vals = []
        if not part.empty:
          vals.append(self.score.highestTime - ndx[-1])
        sers.append(pd.Series(vals, part.index))
      res = pd.concat(sers, axis=1, sort=True)
      if not multi_index and isinstance(res.index, pd.MultiIndex):
        res = res.droplevel(1)
      res.columns = df.columns
      return res

  def midiPitches(self, multi_index=False):
    '''\tReturn a dataframe of notes and rests as midi pitches. Midi does not
    have a representation for rests, so -1 is used as a placeholder.'''
    key = ('midiPitches', multi_index)
    if key not in self._analyses:
      midiPitches = self._m21ObjectsNoTies().applymap(lambda noteOrRest: -1 if noteOrRest.isRest else noteOrRest.pitch.midi, na_action='ignore')
      if not multi_index and isinstance(midiPitches.index, pd.MultiIndex):
        midiPitches = midiPitches.droplevel(1)
      self._analyses[key] = midiPitches
    return self._analyses[key]

  def _noteRestHelper(self, noteOrRest):
    if noteOrRest.isRest:
      return 'r'
    return noteOrRest.nameWithOctave

  def _combineRests(self, col):
      col = col.dropna()
      return col[(col != 'r') | ((col == 'r') & (col.shift(1) != 'r'))]

  def _combineUnisons(self, col):
      col = col.dropna()
      return col[(col == 'r') | (col != col.shift(1))]

  def notes(self, combine_rests=True, combine_unisons=False):
    '''\tReturn a dataframe of the notes and rests given in American Standard Pitch
    Notation where middle C is C4. Rests are designated with the string "r".

    If `combine_rests` is True (default), non-first consecutive rests will be
    removed, effectively combining consecutive rests in each voice.
    `combine_unisons` works the same way for consecutive attacks on the same
    pitch in a given voice, however, `combine_unisons` defaults to False.'''
    if 'notes' not in self._analyses:
      df = self._m21ObjectsNoTies().applymap(self._noteRestHelper, na_action='ignore')
      self._analyses['notes'] = df
    ret = self._analyses['notes'].copy()
    if combine_rests:
      ret = ret.apply(self._combineRests)
    if combine_unisons:
      ret = ret.apply(self._combineUnisons)
    if isinstance(ret.index, pd.MultiIndex):
      ret = ret.droplevel(1)
    return ret

  def _kernNoteHelper(self, _note):
    '''\tParse a music21 note object into a kern note token.'''
    # TODO: this doesn't seem to be detecting longas in scores. Does m21 just not detect longas in kern files? Test with mei, midi, and xml
    startBracket, endBracket, beaming = '', '', ''
    if hasattr(_note, 'tie') and _note.tie is not None:
      if _note.tie.type == 'start':
        startBracket += '['
      elif _note.tie.type == 'continue':
        endBracket += '_'
      elif _note.tie.type == 'stop':
        endBracket += ']'

    spanners = _note.getSpannerSites()
    for spanner in spanners:
      if 'Slur' in spanner.classes:
        if spanner.isFirst(_note):
          startBracket = '(' + startBracket
        elif spanner.isLast(_note):
          endBracket += ')'

    beams = _note.beams.beamsList
    for beam in beams:
      if beam.type == 'start':
        beaming += 'L'
      elif beam.type == 'stop':
        beaming += 'J'

    dur = _duration2Kern[round(float(_note.quarterLength), 5)]
    _oct = _note.octave
    if _oct > 3:
      letter = _note.step.lower() * (_oct - 3)
    else:
      letter = _note.step * (4 - _oct)
    acc = _note.pitch.accidental
    acc = acc.modifier if acc is not None else ''
    longa = 'l' if _note.duration.type == 'longa' else ''
    grace = '' if _note.sortTuple()[4] else 'q'
    return f'{startBracket}{dur}{letter}{acc}{longa}{grace}{beaming}{endBracket}'

  def _kernChordHelper(self, _chord):
    '''\tParse a music21 chord object into a kern chord token.'''
    pitches = []
    dur = _duration2Kern[round(float(_chord.quarterLength), 5)]
    for i, _pitch in enumerate(_chord.pitches):
      _oct = _pitch.octave
      if _oct > 3:
        letter = _pitch.step.lower() * (_oct - 3)
      else:
        letter = _pitch.step * (4 - _oct)
      acc = _pitch.accidental
      acc = acc.modifier if acc is not None else ''
      longa = '' #'l' if _pitch.duration.type == 'longa' else ''
      grace = '' if _chord.sortTuple()[4] else 'q'
      if i == 0:  # beaming and slurs are only on the chord object, so just look for them on the chord for the first pitch
        startBracket, endBracket, beaming = '', '', ''
        spanners = _chord.getSpannerSites()
        for spanner in spanners:
          if 'Slur' in spanner.classes:
            if spanner.isFirst(_chord):
              startBracket = '(' + startBracket
            elif spanner.isLast(_chord):
              endBracket += ')'
        beams = _chord.beams.beamsList
        for beam in beams:
          if beam.type == 'start':
            beaming += 'L'
          elif beam.type == 'stop':
            beaming += 'J'
        pitches.append(f'{startBracket}{dur}{letter}{acc}{longa}{grace}{beaming}{endBracket}')
      else:
        pitches.append(f'{dur}{letter}{acc}{longa}{grace}')
    if len(pitches):
      return ' '.join(pitches)
    else:
      return ''

  def _kernNRCHelper(self, nrc):
    '''\tConvert a music21 note, rest, or chord object to its corresponding kern token.'''
    if nrc.isNote:
      return self._kernNoteHelper(nrc)
    elif nrc.isRest:
      return f'{_duration2Kern.get(round(float(nrc.quarterLength), 5))}r'
    else:
      return self._kernChordHelper(nrc)

  def kernNotes(self):
    '''\tReturn a dataframe of the notes and rests given in kern notation. This is
    not the same as creating a kern format of a score, but is an important step
    in that process.'''
    if 'kernNotes' not in self._analyses:
      # parts = self._parts(multi_index=True)
      sers = []
      divisiStarts = pd.DataFrame(columns=self.partNames)
      divisiEnds = pd.DataFrame(columns=self.partNames)
      for ii, flat_part in enumerate(self._semiFlatParts):
        voces = []
        partName = self.partNames[ii]
        for jj, vox in enumerate(flat_part.voicesToParts()):
          flat = vox.flatten()
          tieBreakers = []
          ser = pd.Series(vox.flatten().getElementsByClass(['Note', 'Rest', 'Chord']), name=self.partNames[ii])
          ser.index = ser.apply(lambda nrc: nrc.offset).astype(float).round(5)
          nexts = ser.index.to_series().shift(-1)
          for kk in range(-1, -1 - len(ser.index), -1):  # reversed to make the *last* event at each offset have a tieBreaker value of 0
            # tieBreakers are the multiIndex values to handle zero-duration grace notes and non-explicit divisi
            if ser.index[kk] == nexts.iat[kk]:
              tieBreakers.append(tieBreakers[-1] - 1)
            else:
              tieBreakers.append(0)
          tieBreakers.reverse()
          dur = ser.apply(lambda nrc: nrc.quarterLength)
          if jj > 0:  # create divisi records (*^ and *v) when looking at a non-first voice in part
            starts = -dur + dur.index
            ends = dur + dur.index
            for val in dur.index:
              if val not in ends.values:
                divisiStarts.at[val, partName] = '*^'
            for val in ends:
              if val not in dur:
                divisiEnds.at[val, partName] = '*v'
          # TODO: pick up here to split things into voices even when they're not declared as separate voices
          # tb = pd.Series(tieBreakers, index=ser.index)
          # d1 = pd.concat([ser, tb, dur], axis=1)
          # mask = (d1.dur > 0) & (d1.tb < 0)
          # d2 = d1[mask]
          # d3 = d1[~mask]
          # d4 = d2[d2.tb < 0]
          # d5 = d1.loc[d4.index] -> if there are any grace notes with tb < -1, also keep them in the first voice and keep searching for more in a loop
          # ***OR*** use flat_part.getOverlaps() and iterate through each list in that dictionary and maybe stream.insertAndShift(element)
          # pdb.set_trace()
          ser.index = pd.MultiIndex.from_arrays((ser.index, tieBreakers))
          ser.name = partName + f'_divisi_{jj}' if jj > 0 else partName
          # pdb.set_trace()
          sers.append(ser)
      df = pd.concat(sers, axis=1)
      df = df.applymap(self._kernNRCHelper, na_action='ignore')
      self._analyses['_divisiStarts'] = divisiStarts.fillna('*')
      self._analyses['_divisiEnds'] = divisiEnds.fillna('*')
      self._analyses['kernNotes'] = df
    return self._analyses['kernNotes'].copy()

  def nmats(self, bpm=60):
    '''\tReturn a dictionary of dataframes, one for each voice, each with the following
    columns about the notes and rests in that voice:

    MEASURE    ONSET_BEAT    DURATION_BEAT    PART    MIDI    ONSET_SEC    OFFSET_SEC

    In the MIDI column, notes are represented with their midi pitch numbers 0 to 127
    inclusive, and rests are represented with -1s. The ONSET and OFFSET columns given
    in seconds are directly proportional to the ONSET_BEATS column and ONSET_BEATS +
    DURATION_BEATS columns respectively. The proportion used is determined by the `bpm`
    argument.'''
    key = ('nmats', bpm)
    if key not in self._analyses:
      nmats = {}
      dur = self.durations()
      dur = dur[~dur.index.duplicated(keep='last')].ffill()  # remove non-last offset repeats and forward-fill
      mp = self.midiPitches()
      mp = mp[~mp.index.duplicated(keep='last')].ffill()  # remove non-last offset repeats and forward-fill
      ms = self._measures()
      toSeconds = 60/bpm
      for i, partName in enumerate(self.partNames):
        meas = ms.iloc[:, i]
        midi = mp.iloc[:, i].dropna()
        onsetBeat = midi.index.to_series()
        durBeat = dur.iloc[:, i].dropna()
        part = pd.Series(partName, midi.index)
        onsetSec = onsetBeat * toSeconds
        offsetSec = (onsetBeat + durBeat) * toSeconds
        df = pd.concat([meas, onsetBeat, durBeat, part, midi, onsetSec, offsetSec], axis=1, sort=True)
        df.columns = ['MEASURE', 'ONSET_BEAT', 'DURATION_BEAT', 'PART', 'MIDI', 'ONSET_SEC', 'OFFSET_SEC']
        df.MEASURE.ffill(inplace=True)
        nmats[partName] = df.dropna()
      self._analyses[key] = nmats
    return self._analyses[key]

  def pianoRoll(self):
    '''\tConstruct midi piano roll. NB: there are 128 possible midi pitches.'''
    if 'pianoRoll' not in self._analyses:
      mp = self.midiPitches()
      mp = mp[~mp.index.duplicated(keep='last')].ffill()  # remove non-last offset repeats and forward-fill
      pianoRoll = pd.DataFrame(index=range(128), columns=mp.index.values)
      for offset in mp.index:
        for pitch in mp.loc[offset]:
          if pitch >= 0:
            pianoRoll.at[pitch, offset] = 1
      pianoRoll.fillna(0, inplace=True)
      self._analyses['pianoRoll'] = pianoRoll
    return self._analyses['pianoRoll']

  def sampled(self, bpm=60, obs=20):
    '''\tSample the score according to bpm, and the desired observations per second, `obs`.'''
    key = ('sampled', bpm, obs)
    if key not in self._analyses:
      slices = 60/bpm * obs
      timepoints = pd.Index([t/slices for t in range(0, int(self.score.highestTime * slices))])
      pr = self.pianoRoll().copy()
      pr.columns = [col if col in timepoints else timepoints.asof(col) for col in pr.columns]
      sampled = pr.reindex(columns=timepoints, method='ffill')
      self._analyses[key] = sampled
    return self._analyses[key]

  def mask(self, winms=100, sample_rate=2000, num_harmonics=1, width=0,
          bpm=60, aFreq=440, base_note=0, tuning_factor=1, obs=20):
    '''\tConstruct a mask from the sampled piano roll using width and harmonics.'''
    key = ('mask', winms, sample_rate, num_harmonics, width, bpm, aFreq, base_note, tuning_factor)
    if key not in self._analyses:
      width_semitone_factor = 2 ** ((width / 2) / 12)
      sampled = self.sampled(bpm, obs)
      num_rows = int(2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
      mask = pd.DataFrame(index=range(num_rows), columns=sampled.columns).fillna(0)
      fftlen = 2**round(math.log(winms / 1000 * sample_rate) / math.log(2))

      for row in range(base_note, sampled.shape[0]):
        note = base_note + row
        # MIDI note to Hz: MIDI 69 = 440 Hz = A4
        freq = tuning_factor * (2 ** (note / 12)) * aFreq / (2 ** (69 / 12))
        if sampled.loc[row, :].sum() > 0:
          mcol = pd.Series(0, index=range(num_rows))
          for harm in range(1, num_harmonics + 1):
            minbin = math.floor(harm * freq / width_semitone_factor / sample_rate * fftlen)
            maxbin = math.ceil(harm * freq * width_semitone_factor / sample_rate * fftlen)
            if minbin <= num_rows:
              maxbin = min(maxbin, num_rows)
              mcol.loc[minbin : maxbin] = 1
          mask.iloc[np.where(mcol)[0], np.where(sampled.iloc[row])[0]] = 1
      self._analyses[key] = mask
    return self._analyses[key]

  def fromJSON(self, json_path):
    '''\tReturn a pandas dataframe of the JSON file. The outermost keys will get
    interpretted as the index values of the table and should be in seconds with
    decimal places allowed, and the second-level keys will be the columns.'''
    with open(json_path) as json_data:
      data = json.load(json_data)
    df = pd.DataFrame(data).T
    df.index = df.index.astype(float)
    return df
  
  def _kernHeader(self):
    '''\tReturn a string of the kern format header global comments.'''
    data = [
      f'!!!COM: {self.metadata["Composer"] or "Composer not found"}',
      f'!!!OTL: {self.metadata["Title"] or "Title not found"}'
    ]
    return '\n'.join(data)
    # f'!!!voices: {len(cols)}', 
    # ['**kern'] * len(cols),

  def _kernFooter(self):
    '''Return a string of the kern format footer global comments.'''
    from datetime import datetime
    data = [
      '!!!RDF**kern: %=rational rhythm',
      '!!!RDF**kern: l=long note in original notation',
      '!!!RDF**kern: i=editorial accidental',
      f'!!!ONB: Translated from a {self.fileExtension} file on {datetime.today().strftime("%Y-%m-%d")} via AMPACT'
    ]
    if 'Title' in self.metadata:
      data.append('!!!title: @{OTL}')
    return '\n'.join(data)

  def toKern(self, path_name='', data='', lyrics=True, dynamics=True):
    '''\t*** WIP: currently not outputting valid kern files. ***
    Create a kern representation of the score. If no `path_name` variable is
    passed, then returns a pandas DataFrame of the kern representation. Otherwise
    a file is created or overwritten at the `path_name` path. If path_name does not
    end in '.krn' then this file extension will be added to the path.
    If `lyrics` is `True` (default) then the lyrics for each part will be added to
    the output, if there are lyrics. The same applies to `dynamics`'''
    key = ('toKern', data)
    if key not in self._analyses:
      _me = self._measures()
      me = _me.astype('string').applymap(lambda cell: '=' + cell + '-' if cell == '0' else '=' + cell, na_action='ignore')
      events = self.kernNotes()
      isMI = isinstance(events.index, pd.MultiIndex)
      includeLyrics, includeDynamics = False, False
      if lyrics and not self.lyrics().empty:
        includeLyrics = True
        lyr = self.lyrics()
        if isMI:
          lyr.index = pd.MultiIndex.from_arrays((lyr.index, [0]*len(lyr.index)))
      if dynamics and not self.dynamics().empty:
        includeDynamics = True
        dyn = self.dynamics()
        if isMI:
          dyn.index = pd.MultiIndex.from_arrays((dyn.index, [0]*len(dyn.index)))
      _cols, firstTokens, partNumbers, staves, instruments, partNames, shortNames = [], [], [], [], [], [], []
      for i in range(len(events.columns), 0, -1):   # reverse column order because kern order is lowest staves on the left
        col = events.columns[i - 1]
        _cols.append(events[col])
        firstTokens.append('**kern')
        partNumbers.append(f'*part{i}')
        staves.append(f'*staff{i}')
        instruments.append('*Ivox')
        partNames.append(f'*I"{col}')
        shortNames.append(f"*I'{col[0]}")
        if includeLyrics and col in lyr.columns:
          _cols.append(lyr[col])
          firstTokens.append('**text')
          partNumbers.append(f'*part{i}')
          staves.append(f'*staff{i}')
          instruments.append('*')
          partNames.append('*')
          shortNames.append('*')
        if includeDynamics and col in dyn.columns:
          _cols.append(dyn[col])
          firstTokens.append('**dynam')
          partNumbers.append(f'*part{i}')
          staves.append(f'*staff{i}')
          instruments.append('*')
          partNames.append('*')
          shortNames.append('*')
      events = pd.concat(_cols, axis=1)
      ba = self._barlines()
      ba = ba[ba != 'regular'].dropna().replace({'double': '||', 'final': '=='})
      ba.loc[self.score.highestTime, :] = '=='
      if isinstance(events.index, pd.MultiIndex):
        events = events.droplevel(1)
      if data:
        cdata = self.fromJSON(data)
        firstTokens.extend(['**data'] * len(cdata.columns))
        partNumbers.extend(['*'] * len(cdata.columns))
        staves.extend(['*'] * len(cdata.columns))
        instruments.extend(['*'] * len(cdata.columns))
        partNames.extend([f'*{col}' for col in cdata.columns])
        shortNames.extend(['*'] * len(cdata.columns))
        events = events[~events.index.duplicated(keep='last')].ffill()  # remove non-last offset repeats and forward-fill
        pdb.set_trace()
        events = pd.concat([events, cdata], axis=1)
      me = pd.concat([me.iloc[:, 0]] * len(events.columns), axis=1)
      ba = pd.concat([ba.iloc[:, 0]] * len(events.columns), axis=1)
      me.columns = events.columns
      ba.columns = events.columns
      ds = self._analyses['_divisiStarts']
      ds = ds.reindex(events.columns, axis=1).fillna('*')
      de = self._analyses['_divisiEnds']
      de = de.reindex(events.columns, axis=1).fillna('*')
      clefs = self._clefs()
      clefs = clefs.reindex(events.columns, axis=1).fillna('*')
      ts = '*M' + self._timeSignatures()
      ts = ts.reindex(events.columns, axis=1).fillna('*')
      ks = self._keySignatures()
      ks = ks.reindex(events.columns, axis=1).fillna('*')
      partTokens = pd.DataFrame([firstTokens, partNumbers, staves, instruments, partNames, shortNames, ['*-']*len(events.columns)],
                                index=[-12, -11, -10, -9, -8, -7, int(self.score.highestTime + 1)])
      partTokens.columns = events.columns
      body = pd.concat([partTokens, de, me, ds, clefs, ks, ts, events, ba]).sort_index(kind='mergesort')
      body = body.fillna('.')
      divRows, divCols = np.where(body == '*^')
      for ii, rowIndex in enumerate(divRows):
        colIndex = divCols[ii]
        colName = body.columns[colIndex]
        targetCols = [jj for jj, col in enumerate(body.columns) if col.startswith(colName) and col != colName]
        if ii == 0:  # delete everying in target cols up to first divisi
          body.iloc[:rowIndex + 1, targetCols] = np.nan
        elif ii + 1 < len(divRows):  # delete everything from the last divisi consolidation to this new divisi
          prevConsolidation = np.where(body.iloc[:rowIndex, colIndex] == '*v')[-1]
          body.iloc[prevConsolidation + 1:rowIndex + 1, targetCols] = np.nan
          body.iloc[prevConsolidation, targetCols] = '*v'
        if ii + 1 == len(divRows):  # delete everything in target cols after final consolidation
          finalConsolidation = np.where(body.iloc[rowIndex:, colIndex] == '*v')[0][0] + rowIndex
          body.iloc[finalConsolidation + 1:, targetCols] = np.nan
          body.iloc[finalConsolidation, targetCols] = '*v'
      result = [self._kernHeader()]
      result.extend(body.apply(lambda row: '\t'.join(row.dropna().astype(str)), axis=1))
      result.extend((self._kernFooter(),))
      result = '\n'.join(result)
      self._analyses[key] = result
    if not path_name:
      return self._analyses[key]
    else:
      if not path_name.endswith('.krn'):
        path_name += '.krn'
      if '/' not in path_name:
        path_name = './output_files/' + path_name
      with open(path_name, 'w') as f:
        f.write(self._analyses[key])
