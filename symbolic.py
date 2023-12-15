import pandas as pd
import numpy as np
import music21 as m21
import math
import ast
import json
import requests
import os
import tempfile
import re
import xml.etree.ElementTree as ET
import base64
import copy
m21.environment.set('autoDownload', 'allow')


_function_pattern = re.compile('[^TtPpDd]')
_volpiano_pattern = re.compile(r'^\d--[a-zA-Z0-9\-\)\?]*$')
_tinyNotation_pattern = re.compile("^[-0-9a-zA-Zn _/'#:~.{}=]+$")
_meiDeclaration = '''<?xml version="1.0" encoding="UTF-8"?>
<!-- <?xml-model href="../../Documents/music-encoding/dist/schemata/mei-all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>-->
<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>
'''
_imported_scores = {}

def _id_gen(start=1):
    """
    Generate a unique ID for each instance of the Score class.

    This function generates a unique ID for each instance of the Score class 
    by incrementing a counter starting from the provided start value. The ID 
    is in the format 'pyAMPACT-{start}'. This isn't meant to be used directly
    so see the example below for usage.

    :param start: An integer representing the starting value for the ID 
        counter. Default is 1.
    :yield: A string representing the unique ID.

    See Also
    --------
    :meth:`insertAudioAnalysis`
    :meth:`xmlIDs`

    Example
    --------
    .. code-block:: python

        newID = next(_idGen)
    """
    while True:
        yield f'pyAMPACT-{start}'
        start += 1
_idGen = _id_gen()

def _remove_namespaces(doc):
    """
    Indent an MEI (Music Encoding Initiative) element for better readability.

    This function recursively indents an MEI element and its children, improving 
    the readability of the MEI XML structure. It modifies the input element in-place.

    :param elem: An xml.etree.ElementTree.Element representing the MEI element.
    :param level: An integer representing the current indentation level. Default is 0.
    :return: None
    """
    root = doc.getroot()
    namespace = ''
    if '}' in root.tag:
        namespace = root.tag[1:root.tag.index('}')]
    for elem in doc.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag[elem.tag.index('}') + 1:]
    if namespace:
        root.set('xmlns', namespace)

def _indentMEI(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indentMEI(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def _insertScoreDef(root, part_names=[]):
    """
    Insert a scoreDef element into an MEI (Music Encoding Initiative) document.

    This function inserts a scoreDef element into an MEI document if one is
    not already present. It modifies the input element in-place.

    :param root: An xml.etree.ElementTree.Element representing the root of the
        MEI document.
    :param part_names: A list of strings representing the names of the parts in
        the score. Default is an empty list.
    :return: None
    """
    if root.find('.//scoreDef') is None:
        scoreDef = ET.Element('scoreDef', {'xml:id': next(_idGen), 'n': '1'})
        if len(part_names) == 0:
            part_names = sorted({f'Part-{staff.attrib.get("n")}' for staff in root.iter('staff')})
        for i, staff in enumerate(part_names):
            staffDef = ET.SubElement(scoreDef, 'staffDef', {'label': staff, 'n': str(i + 1), 'xml:id': next(_idGen)})
            ET.SubElement(staffDef, 'label', {'text': staff, 'xml:id': next(_idGen)})
        scoreEl = root.find('.//score')
        if scoreEl is not None:
            scoreEl.insert(0, scoreDef)

_duration2Kern = {  # keys get rounded to 5 decimal places
    56:      '000..',
    48:      '000.',
    32:      '000',
    28:      '00..',
    24:      '00.',
    16:      '00',
    14:      '0..',
    12:      '0.',
    8:       '0',
    7:       '1..',
    6:       '1.',
    4:       '1',
    3.5:     '2..',
    3:       '2.',
    2.66666: '3%2',
    2:       '2',
    1.75:    '4..',
    1.5:     '4.',
    1.33333: '3',
    1:       '4',
    .875:    '8..',
    .75:     '8.',
    .66667:  '6',
    .5:      '8',
    .4375:   '16..',
    .375:    '16.',
    .33333:  '12',
    .25:     '16',
    .21875:  '32..',
    .1875:   '32.',
    .16667:  '24',
    .125:    '32',
    .10938:  '64..',
    .09375:  '64.',
    .08333:  '48',
    .0625:   '64',
    .05469:  '128..',
    .04688:  '128.',
    .04167:  '96',
    .03125:  '128',
    .02734:  '256..',
    .02344:  '256.',
    .02083:  '192',
    .01563:  '256',
    .01367:  '512..',
    .01172:  '512.',
    .01042:  '384',
    .00781:  '512',
    .00684:  '1024.',
    .00586:  '1024.',
    .00582:  '768',
    .00391:  '1024',
    0:       ''
}

_reused_docstring =  """
    The methods .harmKeys, .harm, .functions, .chords, and .cdata all work in 
    the following way. They get the desired analysis from the relevant spine if 
    this piece is a kern file and has that spine. The default is for the results 
    to be returned as a 1-d array, but you can set `output='series'` for a pandas 
    series instead. If you want to align these results so that they match the 
    columnar (time) axis of the pianoRoll, sampled, or mask results, you can pass 
    the pianoRoll or mask that you want to align to as the `snap_to` parameter.

    The `sampled` and `mask` dfs often have more observations than the spine 
    contents, so you may want to fill in these new empty slots somehow. The kern 
    format uses '.' as a filler token so you can pass this as the `filler` 
    parameter to fill all the new empty slots with this as well. If you choose 
    some other value, say `filler='_'`, then in addition to filling in the empty 
    slots with underscores, this will also replace the kern '.' observations with 
    '_'. If you want to fill them in with NaN's as pandas usually does, you can 
    pass `filler='nan'` as a convenience. If you want to "forward fill" these 
    results, you can pass `filler='forward'` (default). This will propagate the 
    last non-period ('.') observation until a new one is found. Finally, you can 
    pass filler='drop' to drop all empty observations (both NaNs and humdrum
    periods).

    :param snap_to: A pandas DataFrame to align the results to. Default is None.
    :param filler: A string representing the filler token. Default is 'forward'.
    :param output: A string representing the output format. Default is 'array'.
    :return: A numpy array or pandas Series representing the harmonic keys
        analysis.

    See Also
    --------
    :meth:`cdata`
    :meth:`chords`
    :meth:`functions`
    :meth:`harm`
    :meth:`harmKeys`

    Example
    --------
    .. code-block:: python

        piece = Score('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
        piece.harm()
    """


class Score:
    """
    A class to import a musical score via music21 and expose AMPACT's analysis 
    utilities.

    The analysis utilities are generally formatted as pandas dataframes. This 
    class also ports over some matlab code to help with alignment of scores in 
    symbolic notation and audio analysis of recordings of those scores. `Score` 
    objects can insert analysis into an MEI file, and can export any type of 
    file to a kern format, optionally also including analysis from a JSON file. 
    Similarly, `Score` objects can serve clickable URLs of short excerpts of 
    their associated score in symbolic notation. These links open in the Verovio 
    Humdrum Viewer.

    :param score_path: A string representing the path to the score file.
    :return: A Score object.

    Example
    --------
    .. code-block:: python

        url_or_path = 'https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn'
        piece = Score(url_or_path)
    """
    def __init__(self, score_path):
        self._analyses = {}
        if score_path.startswith('https://github.com/'):
            score_path = 'https://raw.githubusercontent.com/' + score_path[19:].replace('/blob/', '/')
        self.path = score_path
        self.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
        self.fileExtension = score_path.rsplit('.', 1)[1] if '.' in score_path else ''
        self._meiTree = None
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
        self._partList()
    
    def _assignM21Attributes(self, path=''):
        """
        Assign music21 attributes to a given object.

        :param obj: A music21 object.
        :return: None
        """
        if self.path not in _imported_scores:
            if path:   # parse humdrum files differently to extract their function, and harm spines if they have them
                _imported_scores[self.path] = m21.converter.parse(path, format='humdrum')
            elif self.fileExtension in ('xml', 'musicxml', 'mei', 'mxl'):   # these files might be mei files and could lack elements music21 needs to be able to read them
                tree = ET.parse(self.path)
                _remove_namespaces(tree)
                root = tree.getroot()
                hasFunctions = False
                _functions = root.findall('.//function')
                if len(_functions):
                    hasFunctions = True

                if root.tag.endswith('mei'):   # this is an mei file even if the fileExtension is .xml
                    parseEdited = False
                    self._meiTree = copy.deepcopy(root)
                    if not root.find('.//scoreDef'):   # this mei file doesn't have a scoreDef element, so construct one and add it to the score
                        parseEdited = True
                        _insertScoreDef(root)

                    for section in root.iter('section'):   # make sure all events are contained in measures
                        if section.find('measure') is None:
                            parseEdited = True
                            measure = ET.Element('measure')
                            measure.set('xml:id', next(_idGen))
                            measure.extend(section)
                            section.clear()
                            section.append(measure)

                    if parseEdited:
                        mei_string = ET.tostring(root, encoding='unicode')
                        _imported_scores[self.path] = m21.converter.subConverters.ConverterMEI().parseData(mei_string)
                        parseEdited = False

                if hasFunctions:   # not an mei file, but an xml file that had functions
                    try:
                        _imported_scores[self.path] = m21.converter.parse(self.path)
                    except m21.harmony.HarmonyException:
                        print('There was an issue with the function texts so they were removed.')
                        for _function in _functions:
                            _function.text = ''
                        xml_string = ET.tostring(root, encoding='unicode')
                        _imported_scores[self.path] = m21.converter.parse(xml_string, format='MusicXML')

            elif self.fileExtension in ('', 'txt'):   # read file/string as volpiano or tinyNotation if applicable
                temp = None
                text = self.path
                if self.fileExtension == 'txt':
                    with open(self.path, 'r') as file:
                        text = file.read()
                if text.startswith('volpiano: ') or re.match(_volpiano_pattern, text):
                    temp = m21.converter.parse(text, format='volpiano')
                elif text.startswith('tinyNotation: ') or re.match(_tinyNotation_pattern, text):
                    temp = m21.converter.parse(text, format='tinyNotation')
                if temp is not None:
                    _score = m21.stream.Score()
                    _score.insert(0, temp)
                    _imported_scores[self.path] = _score

        if self.path not in _imported_scores:   # check again to catch valid tree files
            _imported_scores[self.path] = m21.converter.parse(self.path)
        self.score = _imported_scores[self.path]
        self.metadata = {'title': "Title not found", 'composer': "Composer not found"}
        if self.score.metadata is not None:
            self.metadata['title'] = self.score.metadata.title or 'Title not found'
            self.metadata['composer'] = self.score.metadata.composer or 'Composer not found'
        self._partStreams = self.score.getElementsByClass(m21.stream.Part)
        self._flatParts = []
        self.partNames = []
        for i, part in enumerate(self._partStreams):
            flat = part.flatten()
            toRemove = [el for el in flat if el.offset < 0]
            flat.remove(toRemove)
            flat.makeMeasures(inPlace=True)
            self._flatParts.append(flat.flatten())   # you have to flatten again after calling makeMeasures
            name = flat.partName if (flat.partName and flat.partName not in self.partNames) else 'Part-' + str(i + 1)
            self.partNames.append(name)

    def _addTieBreakers(self, partList):
        """
        Add tie-breaker level to index. Changes parts in partList in place and 
        returns None. 

        This is particularly useful to disambiguate the order of events that 
        happen at the same offset, which is an issue most commonly encountered 
        with grace notes since they have no duration. This is needed in several 
        `Score` methods because you cannot append multiple pandas series (parts) 
        if they have non-unique indices. So this method is needed internally to 
        be able to use pd.concat to turn a list of series into a single dataframe 
        if any of those series has a repeated value in its index.

        :param partList: A list of pandas Series, each representing a part in 
            the score.
        :return: None
        """
        for part in partList:
            if isinstance(part.index, pd.MultiIndex):
                continue
            tieBreakers = []
            nexts = part.index.to_series().shift(-1)
            for ii in range(-1, -1 - len(part.index), -1):
                if part.index[ii] == nexts.iat[ii]:
                    tieBreakers.append(tieBreakers[-1] - 1)
                else:
                    tieBreakers.append(0)
            tieBreakers.reverse()
            part.index = pd.MultiIndex.from_arrays((part.index, tieBreakers))

    def _partList(self):
        """
        Return a list of series of the note, rest, and chord objects in each part.

        :return: A list of pandas Series, each representing a part in the score.
        """
        if '_partList' not in self._analyses:
            kernStrands = []
            parts = []
            isUnique = True
            divisiStarts = []
            divisiEnds = []
            for ii, flat_part in enumerate(self._flatParts):
                graces, graceOffsets = [], []
                notGraces = {}
                for nrc in flat_part.getElementsByClass(['Note', 'Rest', 'Chord']):
                    if nrc.sortTuple()[4]:
                        if (nrc.isRest and nrc.quarterLength > 18):  # get rid of really long rests TODO: make this get rid of rests longer than the prevailing measure
                            continue
                        offset = round(float(nrc.offset), 5)
                        if offset in notGraces:
                            notGraces[offset].append(nrc)
                        else:
                            notGraces[offset] = [nrc]
                    else:
                        graces.append(nrc)
                        graceOffsets.append(round(float(nrc.offset), 5))
                    
                ser = pd.Series(notGraces)
                if ser.empty:   # no note, rest, or chord objects detected in this part
                    ser.name = self.partNames[ii]
                    parts.append(ser)
                    continue
                df = ser.apply(pd.Series)  # make each cell a row resulting in a df where each col is a separate synthetic voice
                if len(df.columns > 1):  # swap elements in cols at this offset until all of them fill the space left before the next note in each col
                    for jj, ndx in enumerate(df.index):
                        # calculate dur inside the loop to avoid having to swap its elements like we do for df
                        dur = df.applymap(lambda cell: round(float(cell.quarterLength), 5), na_action='ignore')
                        for thisCol in range(len(df.columns) - 1):
                            if isinstance(df.iat[jj, thisCol], float):  # ignore NaNs
                                continue
                            thisDur = dur.iat[jj, thisCol]
                            thisNextNdx = df.iloc[jj+1:, thisCol].first_valid_index() or self.score.highestTime
                            thisPrevNdx = df.iloc[:jj, thisCol].last_valid_index() or 0
                            if thisPrevNdx > 0:
                                thisPrevDur = dur[thisCol].at[thisPrevNdx]
                                if thisPrevNdx + thisPrevDur - ndx > .00003:    # current note happens before previous note ended so swap for a NaN if there is one
                                    for otherCol in range(thisCol + 1, len(df.columns)):
                                        if isinstance(df.iat[jj, otherCol], float):
                                            df.iloc[jj, [thisCol, otherCol]] = df.iloc[jj, [otherCol, thisCol]]
                                            break
                            if abs(thisNextNdx - ndx - thisDur) < .00003:   # this nrc takes up the amount of time expected in this col so no need to swap
                                continue
                            for otherCol in range(thisCol + 1, len(df.columns)):  # look for an nrc in another col with the duration thisCol needs
                                if isinstance(df.iat[jj, otherCol], float):  # once we get a nan there's no hope of finding a valid swap at this index
                                    break
                                otherDur = dur.iat[jj, otherCol]
                                if abs(thisNextNdx - ndx - otherDur) < .00003:  # found a valid swap
                                    df.iloc[jj, [thisCol, otherCol]] = df.iloc[jj, [otherCol, thisCol]]
                                    break

                if len(graces):  # add all the grace notes found to col0
                    part0 = pd.concat((pd.Series(graces, graceOffsets), df.iloc[:, 0].dropna())).sort_index(kind='mergesort')
                    isUnique = False
                else:
                    part0 = df.iloc[:, 0].dropna()
                part0.name = self.partNames[ii]
                parts.append(part0)
                kernStrands.append(part0)

                strands = []
                for col in range(1, len(df.columns)):  # if df has more than 1 column, iterate over the non-first columns
                    part = df.iloc[:, col].dropna()
                    _copy = part.copy()
                    _copy.name = f'{part0.name}_{col}'
                    parts.append(_copy)
                    dur = part.apply(lambda nrc: nrc.quarterLength).astype(float).round(5)
                    prevEnds = (dur + dur.index).shift()
                    startI = 0
                    for endI, endNdx in enumerate(part.index[startI:]):
                        endNdx = round(float(endNdx), 5)
                        nextNdx = self.score.highestTime if len(part) - 1 == endI else part.index[endI + 1]
                        thisDur = part.iat[endI].quarterLength
                        if abs(nextNdx - endNdx - thisDur) > .00003:
                            strand = part.iloc[startI:endI + 1].copy()
                            strand.name = f'{self.partNames[ii]}__{len(strands) + 1}'
                            divisiStarts.append(pd.Series(('*^', '*^'), index=(strand.name, self.partNames[ii]), name=part.index[startI]))
                            joinNdx = endNdx + thisDur        # find a suitable endpoint to rejoin this strand
                            divisiEnds.append(pd.Series(('*v', '*v'), index=(strand.name, self.partNames[ii]), name=(strand.name, joinNdx)))
                            strands.append(strand)
                            startI = endI + 1
                kernStrands.extend(sorted(strands, key=lambda _strand: _strand.last_valid_index()))

            self._analyses['_divisiStarts'] = pd.DataFrame(divisiStarts).fillna('*').sort_index()
            de = pd.DataFrame(divisiEnds)
            if not de.empty:
                de = de.reset_index(level=1)
                de = de.reindex([prt.name for prt in kernStrands if prt.name not in self.partNames]).set_index('level_1')
            self._analyses['_divisiEnds'] = de
            if not isUnique:
                self._addTieBreakers(parts)
                self._addTieBreakers(kernStrands)
            self._analyses['_partList'] = parts
            self._analyses['_kernStrands'] = kernStrands
        return self._analyses['_partList']

    def _parts(self, multi_index=False, kernStrands=False, compact=False,
            number=False):
        """
        Return a DataFrame of the note, rest, and chord objects in the score.

        The difference between parts and kernStrands is that parts can have voices
        whereas kernStrands cannot. If there are voices in the _parts DataFrame, the
        kernStrands DataFrame will include all these notes by adding additional
        columns.

        :param multi_index: Boolean, default False. If True, the returned DataFrame
            will have a MultiIndex.
        :param kernStrands: Boolean, default False. If True, the method will use the
            '_kernStrands' analysis.
        :param compact: Boolean, default False. If True, the method will keep chords
            unified rather then expanding them into separate columns.
        :param number: Boolean, default False. If True, the method will 1-index
            the part names and the voice names making the columns a MultiIndex. Only
            applies if `compact` is also True.
        :return: A DataFrame of the note, rest, and chord objects in the score.
        """
        key = ('_parts', multi_index, kernStrands, compact, number)
        if key not in self._analyses:
            toConcat = []
            if kernStrands:
                toConcat = self._analyses['_kernStrands']
            elif compact:
                toConcat = self._analyses['_partList']
                if number:
                    partNameToNum = {part: i + 1 for i, part in enumerate(self.partNames)}
                    colTuples = []
                    for part in toConcat:
                        names = part.name.split('_')
                        if len(names) == 1:
                            colTuples.append((partNameToNum[names[0]], 1))
                        else:
                            colTuples.append((partNameToNum[names[0]], int(names[1]) + 1))
                    mi = pd.MultiIndex.from_tuples(colTuples, names=('Staff', 'Layer'))
            else:
                for part in self._partList():
                    if part.empty:
                        toConcat.append(part)
                        continue
                    listify = part.apply(lambda nrc: nrc.notes if nrc.isChord else [nrc])
                    expanded = listify.apply(pd.Series)
                    expanded.columns = [f'{part.name}:{i}' if i > 0 else part.name for i in range(len(expanded.columns))]
                    toConcat.append(expanded)
            df = pd.concat(toConcat, axis=1, sort=True) if len(toConcat) else pd.DataFrame(columns=self.partNames)
            if not multi_index and isinstance(df.index, pd.MultiIndex):
                df.index = df.index.droplevel(1)
            if compact and number:
                df.columns = mi
            self._analyses[key] = df
        return self._analyses[key]

    def _import_function_harm_spines(self, path=''):
        """
        Import the harmonic function spines from a given path.

        :param path: A string representing the path to the file containing the 
            harmonic function spines.
        :return: A pandas DataFrame representing the harmonic function spines.
        """
        if self.fileExtension == 'krn' or path:
            humFile = m21.humdrum.spineParser.HumdrumFile(path or self.path)
            humFile.parseFilename()
            for spine in humFile.spineCollection:
                if spine.spineType in ('harm', 'function', 'chord', 'cdata'):
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
                                func = _function_pattern.sub('', contents)
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

        for spine in ('function', 'harm', 'harmKeys', 'chord'):
            if spine not in self._analyses:
                self._analyses[spine] = pd.Series()
        if 'cdata' not in self._analyses:
            self._analyses['cdata'] = pd.DataFrame()

    def xmlIDs(self):
        """
        Return xml ids per part in a pandas.DataFrame time-aligned with the
        objects offset. If the file is not xml or mei, or an idString wasn't found,
        return a DataFrame of the ids of the music21 objects.

        :return: A pandas DataFrame representing the xml ids in the score.

        See Also
        --------
        :meth:`nmats`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.xmlIDs()
        """
        if 'xmlIDs' in self._analyses:
            return self._analyses['xmlIDs']
        if self.fileExtension in ('xml', 'mei'):
            tree = ET.parse(self.path)
            root = tree.getroot()
            idString = [key for key in root.attrib.keys() if key.endswith('}id')]
            if len(idString):
                idString = idString[0]
                data = {}
                dotCoefficients = {None: 1, '1': 1.5, '2': 1.75, '3': 1.875, '4': 1.9375}
                for staff in root.findall('.//staff'):
                    for layer in staff.findall('layer'):   # doesn't need './/' because only looks for direct children of staff elements
                        column_name = f"Staff{staff.get('n')}_Layer{layer.get('n')}"
                        if column_name not in data:
                            data[column_name] = []
                        for nrb in layer:
                            if nrb.tag.endswith('note') or nrb.tag.endswith('rest') or nrb.tag.endswith('mRest'):
                                data[column_name].append(nrb.get(idString))
                            elif nrb.tag.endswith('beam'):
                                for nr in nrb:
                                    data[column_name].append(nr.get(idString))
                ids = pd.DataFrame.from_dict(data, orient='index').T
                cols = []
                parts = self._parts(multi_index=True).copy()
                for i in range(len(parts.columns)):
                    part = parts.iloc[:, i].dropna()
                    idCol = ids.iloc[:, i].dropna()
                    idCol.index = part.index
                    cols.append(idCol)
                df = pd.concat(cols, axis=1)
                df.columns = parts.columns
                self._analyses['xmlIDs'] = df
                return df
        # either not xml/mei, or an idString wasn't found
        df = self._parts(multi_index=True).applymap(lambda obj: str(obj.id), na_action='ignore')
        self._analyses['xmlIDs'] = df
        return df

    def lyrics(self):
        """
        Extract the lyrics from the score. 

        The lyrics are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a lyric. The 
        DataFrame is indexed by the offset of the lyrics.

        :return: A pandas DataFrame representing the lyrics in the score.

        See Also
        --------
        :meth:`dynamics`

        Example
        -------
        .. code-block:: python

            piece = Score('https://raw.githubusercontent.com/alexandermorgan/AMPACT/main/test_files/busnoys.krn')
            piece.lyrics()
        """
        if 'lyrics' not in self._analyses:
            self._analyses['lyrics'] = self._parts().applymap(lambda cell: cell.lyric if hasattr(cell, 'lyric') else np.nan, na_action='ignore').dropna(how='all')
        return self._analyses['lyrics'].copy()

    def _clefHelper(self, clef):
        """
        Parse a music21 clef object into the corresponding humdrum syntax token.

        :param clef: A music21 clef object.
        :return: A string representing the humdrum syntax token for the clef.
        """
        octaveChange = ''
        if clef.octaveChange > 0:
            octaveChange = '^' * clef.octaveChange
        elif clef.octaveChange < 0:
            octaveChange = 'v' * abs(clef.octaveChange)
        return f'*clef{clef.sign}{octaveChange}{clef.line}'

    def _clefs(self):
        """
        Extract the clefs from the score. 

        The clefs are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a clef. The 
        DataFrame is indexed by the offset of the clefs.

        :return: A pandas DataFrame representing the clefs in the score.
        """
        if 'clefs' not in self._analyses:
            parts = []
            isUnique = True
            for i, flat_part in enumerate(self._flatParts):
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
        """
        Extract the dynamics from the score. 

        The dynamics are extracted from each part and returned as a pandas DataFrame 
        where each column represents a part and each row represents a dynamic 
        marking. The DataFrame is indexed by the offset of the dynamic markings.

        :return: A pandas DataFrame representing the dynamics in the score.

        See Also
        --------
        :meth:`lyrics`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.dynamics()
        """
        if 'dynamics' not in self._analyses:
            dyns = [pd.Series({obj.offset: obj.value for obj in sf.getElementsByClass('Dynamic')}) for sf in self._flatParts]
            dyns = pd.concat(dyns, axis=1)
            dyns.columns = self.partNames
            dyns.dropna(how='all', axis=1, inplace=True)
            self._analyses['dynamics'] = dyns
        return self._analyses['dynamics'].copy()

    def _priority(self):
        """
        For .krn files, get the line numbers of the events in the piece, which 
        music21 often calls "priority". For other encoding formats return an 
        empty dataframe.

        :return: A DataFrame containing the priority values.
        """
        if '_priority' not in self._analyses:
            if self.fileExtension != 'krn':
                priority = pd.DataFrame()
            else:
                priority = self._parts().applymap(lambda cell: cell.priority, na_action='ignore').ffill(axis=1).iloc[:, -1].astype('Int16')
                priority = pd.DataFrame({'Priority': priority.values, 'Offset': priority.index})
            self._analyses['_priority'] = priority
        return self._analyses['_priority']

    def _snapTo(self, data, snap_to=None, filler='forward', output='array'):
        """"
        Takes a `harm`, `harmKeys`, `functions`, `chords`, or `cdata` as `data` and
        the `snap_to` and `filler` parameters as described in the former three's 
        doc strings.

        The passed data is returned in the shape of the snap_to dataframe's columns,
        and any filling operations are applied. The output will be in the form of a 
        1D numpy array unless `output` is changed, in which case a series will be 
        returned for harm, harmKeys, functions, and chords data, and a dataframe for 
        cdata data.

        :param data: Can be `harm`, `harmKeys`, `functions`, `chords`, or `cdata`.
        :param snap_to: Described in the docstrings of `harm`, `harmKeys`, and 
            `functions`.
        :param filler: Described in the docstrings of `harm`, `harmKeys`, and 
            `functions`.
        :param output: If changed, a series will be returned for `harm`, `harmKeys`, 
            `functions`, and `chords` data, and a dataframe for `cdata` data. Default 
            is a 1D numpy array.
        :return: The passed data in the shape of the `snap_to` dataframe's columns 
            with any filling operations applied.
        """
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
        return self._snapTo(self._analyses['harmKeys'].copy(), snap_to, filler, output)
    harmKeys.__doc__ = _reused_docstring

    def harm(self, snap_to=None, filler='forward', output='array'):
        return self._snapTo(self._analyses['harm'].copy(), snap_to, filler, output)
    harm.__doc__ = _reused_docstring

    def functions(self, snap_to=None, filler='forward', output='array'):
        return self._snapTo(self._analyses['function'].copy(), snap_to, filler, output)
    functions.__doc__ = _reused_docstring

    def chords(self, snap_to=None, filler='forward', output='array'):
        return self._snapTo(self._analyses['chord'].copy(), snap_to, filler, output)
    chords.__doc__ = _reused_docstring

    def cdata(self, snap_to=None, filler='forward', output='dataframe'):
        return self._snapTo(self._analyses['cdata'].copy(), snap_to, filler, output)
    cdata.__doc__ = _reused_docstring

    def _removeTied(self, noteOrRest):
        """
        Helper function for the `_m21ObjectsNoTies` method. 

        Remove tied notes in a given note or rest. Only the first note in a tied 
        group will be kept.

        :param noteOrRest: A music21 note or rest object.
        :return: np.nan if the note is tied and not the first in the group, 
            otherwise the original note or rest.
        """
        if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
            return np.nan
        return noteOrRest

    def _m21ObjectsNoTies(self):
        """
        Remove tied notes in a given voice. Only the first note in a tied group 
        will be kept.

        :param voice: A music21 stream Voice object.
        :return: A list of music21 objects with ties removed.
        """
        if '_m21ObjectsNoTies' not in self._analyses:
            self._analyses['_m21ObjectsNoTies'] = self._parts(multi_index=True).applymap(self._removeTied).dropna(how='all')
        return self._analyses['_m21ObjectsNoTies']

    def _measures(self):
        """
        Return a DataFrame of the measure starting points.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a measure start. The values are 
            the measure numbers.
        """
        if '_measures' not in self._analyses:
            partMeasures = []
            for i, partName in enumerate(self.partNames):
                ser = pd.Series({m.offset: m.measureNumber for m in self._flatParts[i].makeMeasures()}, dtype='Int16')
                partMeasures.extend([ser] * len([part for part in self._parts().columns if part.startswith(partName)]))
            df = pd.concat(partMeasures, axis=1)
            df.columns = self._parts().columns
            self._analyses['_measures'] = df
        return self._analyses['_measures'].copy()

    def _barlines(self):
        """
        Return a DataFrame of barlines specifying which barline type.

        Double barline, for example, can help detect section divisions, and the 
        final barline can help process the `highestTime` similar to music21.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a barline. The values are the 
            barline types.
        """
        if "_barlines" not in self._analyses:
            partBarlines = [pd.Series({m.offset: m.measureNumber for m in part.getElementsByClass(['Barline'])})
                                            for i, part in enumerate(self._flatParts)]
            df = pd.concat(partBarlines, axis=1)
            df.columns = self.partNames
            self._analyses["_barlines"] = df
        return self._analyses["_barlines"]

    def _keySignatures(self, kern=True):
        """
        Return a DataFrame of key signatures for each part in the score.

        :param kern: Boolean, default True. If True, the key signatures are 
            returned in the **kern format.
        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a key signature. The values are 
            the key signatures.
        """
        if '_keySignatures' not in self._analyses:
            kSigs = []
            for i, part in enumerate(self._flatParts):
                kSigs.append(pd.Series({ky.offset: ky for ky in part.getElementsByClass(['Key'])}, name=self.partNames[i]))          
            df = pd.concat(kSigs, axis=1).sort_index(kind='mergesort')
            if kern:
                df = '*k[' + df.applymap(lambda ky: ''.join([_note.name for _note in ky.alteredPitches]).lower(), na_action='ignore') + ']'
            self._analyses['_keySignatures'] = df
        return self._analyses['_keySignatures']

    def _timeSignatures(self):
        """
        Return a DataFrame of time signatures for each part in the score.

        :return: A DataFrame where each column corresponds to a part in the score,
            and each row index is the offset of a time signature. The values are 
            the time signatures in ratio string format.
        """
        if '_timeSignatures' not in self._analyses:
            tsigs = []
            for i, part in enumerate(self._flatParts):
                tsigs.append(pd.Series({ts.offset: ts.ratioString for ts in part.getTimeSignatures()}, name=self.partNames[i]))
            df = pd.concat(tsigs, axis=1).sort_index(kind='mergesort')
            self._analyses['_timeSignatures'] = df
        return self._analyses['_timeSignatures']

    def durations(self, multi_index=False, df=None):
        """
        Return a DataFrame of durations of note and rest objects in the piece.

        If a DataFrame is provided as `df`, the method calculates the difference 
        between cell offsets per column in the passed DataFrame, skipping 
        memoization.

        :param multi_index: Boolean, default False. If True, the returned DataFrame 
            will have a MultiIndex.
        :param df: Optional DataFrame. If provided, the method calculates the 
            difference between cell offsets per column in this DataFrame.
        :return: A DataFrame of durations of note and rest objects in the piece.

        See Also
        --------
        :meth:`notes`
            Return a DataFrame of the notes and rests given in American Standard
            Pitch Notation

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.durations()
        """
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
        """
        Return a DataFrame of notes and rests as MIDI pitches.

        MIDI does not have a representation for rests, so -1 is used as a 
        placeholder.

        :param multi_index: Boolean, default False. If True, the returned DataFrame 
            will have a MultiIndex.
        :return: A DataFrame of notes and rests as MIDI pitches. Rests are 
            represented as -1.

        See Also
        --------
        :meth:`kernNotes`
            Return a DataFrame of the notes and rests given in kern notation.
        :meth:`notes`
            Return a DataFrame of the notes and rests given in American Standard
            Pitch Notation

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.midiPitches()
        """
        key = ('midiPitches', multi_index)
        if key not in self._analyses:
            midiPitches = self._m21ObjectsNoTies().applymap(lambda nr: -1 if nr.isRest else nr.pitch.midi, na_action='ignore')
            if not multi_index and isinstance(midiPitches.index, pd.MultiIndex):
                midiPitches = midiPitches.droplevel(1)
            self._analyses[key] = midiPitches
        return self._analyses[key]

    def _noteRestHelper(self, nr):
        """
        Helper function for the `notes` method. 

        If the note/rest object `nr` is a rest, return 'r'. Otherwise, return the 
        note's name with octave.

        :param nr: A note/rest object.
        :return: 'r' if `nr` is a rest, otherwise the note's name with octave.
        """
        if nr.isRest:
            return 'r'
        return nr.nameWithOctave

    def _combineRests(self, col):
        """
        Helper function for the `notes` method. 

        Combine consecutive rests in a given voice. Non-first consecutive rests 
        will be removed.

        :param col: A pandas Series representing a voice.
        :return: The same pandas Series with consecutive rests combined.
        """
        col = col.dropna()
        return col[(col != 'r') | ((col == 'r') & (col.shift(1) != 'r'))]

    def _combineUnisons(self, col):
        """
        Helper function for the `notes` method. 

        Combine consecutive unisons in a given voice. Non-first consecutive unisons 
        will be removed.

        :param col: A pandas Series representing a voice.
        :return: The same pandas Series with consecutive unisons combined.
        """
        col = col.dropna()
        return col[(col == 'r') | (col != col.shift(1))]

    def notes(self, combine_rests=True, combine_unisons=False):
        """
        Return a DataFrame of the notes and rests given in American Standard Pitch
        Notation where middle C is C4. Rests are designated with the string "r".

        If `combine_rests` is True (default), non-first consecutive rests will be
        removed, effectively combining consecutive rests in each voice.
        `combine_unisons` works the same way for consecutive attacks on the same
        pitch in a given voice, however, `combine_unisons` defaults to False.

        :param combine_rests: Boolean, default True. If True, non-first consecutive 
            rests will be removed.
        :param combine_unisons: Boolean, default False. If True, consecutive attacks 
            on the same pitch in a given voice will be combined.
        :return: A DataFrame of notes and rests in American Standard Pitch Notation.

        See Also
        --------
        :meth:`kernNotes`
            Return a DataFrame of the notes and rests given in kern notation.
        :meth:`midiPitches`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.notes()
        """
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
        """
        Parse a music21 note object into a kern note token.

        This method handles the conversion of various musical notations such as 
        ties, slurs, beams, durations, octaves, accidentals, longas, and grace 
        notes into the kern format.

        :param _note: A music21 note object to be converted into a kern note token.
        :return: A string representing the kern note token.
        """
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
        """
        Parse a music21 chord object into a kern chord token.

        This method uses the `_kernNoteHelper` method to convert each note in the 
        chord into a kern note token. The tokens are then joined together with 
        spaces to form the kern chord token.

        :param _chord: A music21 chord object to be converted into a kern chord token.
        :return: A string representing the kern chord token.
        """
        return ' '.join([self._kernNoteHelper(note) for note in _chord.notes])

    def _kernNRCHelper(self, nrc):
        """
        Convert a music21 note, rest, or chord object to its corresponding kern token.

        This method uses the `_kernNoteHelper` and `_kernChordHelper` methods to 
        convert note and chord objects, respectively. Rest objects are converted 
        directly in this method.

        :param nrc: A music21 note, rest, or chord object to be converted into a 
            kern token.
        :return: A string representing the kern token.
        """
        if nrc.isNote:
            return self._kernNoteHelper(nrc)
        elif nrc.isRest:
            return f'{_duration2Kern.get(round(float(nrc.quarterLength), 5))}r'
        else:
            return self._kernChordHelper(nrc)

    def kernNotes(self):
        """
        Return a DataFrame of the notes and rests given in kern notation.

        This is not the same as creating a kern format of a score, but is an 
        important step in that process.

        :return: A DataFrame of notes and rests in kern notation.

        See Also
        --------
        :meth: `midiPitches`
        :meth:`notes`
            Return a DataFrame of the notes and rests given in American Standard
            Pitch Notation

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.kernNotes()
        """
        if 'kernNotes' not in self._analyses:
            self._analyses['kernNotes'] = self._parts(True, True).applymap(self._kernNRCHelper, na_action='ignore')
        return self._analyses['kernNotes']

    def nmats(self, json_path=None, include_cdata=False):
        """
        Return a dictionary of DataFrames, one for each voice, with information 
        about the notes and rests in that voice.

        Each DataFrame has the following columns:
        
        MEASURE  ONSET  DURATION  PART  MIDI  ONSET_SEC  OFFSET_SEC
        
        In the MIDI column, notes are represented 
        with their MIDI pitch numbers (0 to 127), and rests are represented with -1s. 
        The ONSET_SEC and OFFSET_SEC columns are taken from the audio analysis from 
        the `json_path` file if one is given. The XML_IDs of each note or rest serve 
        as the index for this DataFrame. If `include_cdata` is True and a `json_path` 
        is provided, the cdata from the json file is included in the DataFrame.

        :param json_path: Optional path to a JSON file containing audio analysis data.
        :param include_cdata: Boolean, default False. If True and a `json_path` is 
            provided, the cdata from the json file is included in the DataFrame.
        :return: A dictionary of DataFrames, one for each voice.

        See Also
        --------
        :meth:`fromJSON`
        :meth:`insertAudioAnalysis`
        :meth:`jsonCDATA`
        :meth:`xmlIDs`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K179/Krn/K179.krn')
            piece.nmats()
        """
        if not json_path:   # user must pass a json_path if they want the cdata to be included
            include_cdata = False
        key = ('nmats', json_path, include_cdata)
        if key not in self._analyses:
            nmats = {}
            included = {}
            dur = self.durations(multi_index=True)
            mp = self.midiPitches(multi_index=True)
            ms = self._measures()
            ids = self.xmlIDs()
            data = self.fromJSON(json_path) if json_path else pd.DataFrame()
            if isinstance(ids.index, pd.MultiIndex):
                ms.index = pd.MultiIndex.from_product((ms.index, (0,)))
            for i, partName in enumerate(self._parts().columns):
                meas = ms.iloc[:, i]
                midi = mp.iloc[:, i].dropna()
                onsetBeat = pd.Series(midi.index.get_level_values(0), index = midi.index)
                durBeat = dur.iloc[:, i].dropna()
                part = pd.Series(partName, midi.index)
                xmlID = ids.iloc[:, i].dropna()
                onsetSec = pd.Series()
                offsetSec = pd.Series()
                df = pd.concat([meas, onsetBeat, durBeat, part, midi, onsetSec, offsetSec, xmlID], axis=1, sort=True)
                df.columns = ['MEASURE', 'ONSET', 'DURATION', 'PART', 'MIDI', 'ONSET_SEC', 'OFFSET_SEC', 'XML_ID']
                df.MEASURE.ffill(inplace=True)
                df.dropna(how='all', inplace=True, subset=df.columns[1:5])
                df = df.set_index('XML_ID')
                if json_path is not None:   # add json data if a json_path is provided
                    if len(data.index) > len(df.index):
                        data = data.iloc[:len(df.index), :]
                        print('\n\n*** Warning ***\n\nThe json data has more observations than there are notes in this part so the data was truncated.\n')
                    elif len(data.index) < len(df.index):
                        print('\n\n*** Warning ***\n\nThere are more events than there are json records in this part.\n')
                    df.iloc[:len(data.index), 5] = data.index
                    if len(data.index) > 1:
                        df.iloc[:len(data.index) - 1, 6] = data.index[1:]
                    data.index = df.index[:len(data.index)]
                    df = pd.concat((df, data), axis=1)
                    included[partName] = df
                    df = df.iloc[:, :7].copy()
                nmats[partName] = df
            self._analyses[('nmats', json_path, False)] = nmats
            if json_path:
                self._analyses[('nmats', json_path, True)] = included
        return self._analyses[key]

    def pianoRoll(self):
        """
        Construct a MIDI piano roll.

        Note: There are 128 possible MIDI pitches.

        :return: A DataFrame representing the MIDI piano roll. Each row corresponds 
            to a MIDI pitch (0 to 127), and each column corresponds to an offset in 
            the score. The values are 1 for a note onset and 0 otherwise.

        See Also
        --------
        :meth:`mask`
        :meth:`sampled`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.pianoRoll()
        """
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
        """
        Sample the score according to the given beats per minute (bpm) and the 
        desired observations per second (obs).

        :param bpm: Integer, default 60. The beats per minute to use for sampling.
        :param obs: Integer, default 20. The desired observations per second.
        :return: A DataFrame representing the sampled score. Each row corresponds 
            to a MIDI pitch (0 to 127), and each column corresponds to a timepoint 
            in the sampled score. The values are 1 for a note onset and 0 otherwise.

        See Also
        --------
        :meth:`mask`
        :meth:`pianoRoll`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.sampled()
        """
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
        """
        Construct a mask from the sampled piano roll using width and harmonics.

        :param winms: Integer, default 100. The window size in milliseconds.
        :param sample_rate: Integer, default 2000. The sample rate in Hz.
        :param num_harmonics: Integer, default 1. The number of harmonics to use.
        :param width: Integer, default 0. The width of the mask.
        :param bpm: Integer, default 60. The beats per minute to use for sampling.
        :param aFreq: Integer, default 440. The frequency of A4 in Hz.
        :param base_note: Integer, default 0. The base MIDI note to use.
        :param tuning_factor: Float, default 1. The tuning factor to use.
        :param obs: Integer, default 20. The desired observations per second.
        :return: A DataFrame representing the mask. Each row corresponds to a 
            frequency bin, and each column corresponds to a timepoint in the 
            sampled score. The values are 1 for a note onset and 0 otherwise.

        See Also
        --------
        :meth:`pianoRoll`
        :meth:`sampled`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
            piece.mask()
        """
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

    def jsonCDATA(self, json_path):
        """
        Return a dictionary of pandas DataFrames, one for each voice. These 
        DataFrames contain the cdata from the JSON file designated in `json_path` 
        with each nested key in the JSON object becoming a column name in the 
        DataFrame. The outermost keys of the JSON cdata will become the "absolute" 
        column. While the columns are different, there are as many rows in these 
        DataFrames as there are in those of the nmats DataFrames for each voice.

        :param json_path: Path to a JSON file containing cdata.
        :return: A dictionary of pandas DataFrames, one for each voice.

        See Also
        --------
        :meth:`fromJSON`
        :meth:`insertAudioAnalysis`
        :meth:`nmats`
        :meth:`xmlIDs`

        Example
        -------
        .. code-block:: python

            piece = Score('./test_files/CloseToYou.mei.xml')
            piece.jsonCDATA(json_path='./test_files/CloseToYou.json')
        """
        key = ('jsonCDATA', json_path)
        if key not in self._analyses:
            nmats = self.nmats(json_path=json_path, include_cdata=True)
            cols = ['ONSET_SEC'] + next(iter(nmats.values())).columns[7:].to_list()
            post = {}
            for partName, df in nmats.items():
                res = df[cols].copy()
                res.rename(columns={'ONSET_SEC': 'absolute'}, inplace=True)
                post[partName] = res
            self._analyses[key] = post
        return self._analyses[key]

    def fromJSON(self, json_path):
        """
        Load a JSON file into a pandas DataFrame.

        The outermost keys of the JSON object are interpreted as the index values of 
        the DataFrame and should be in seconds with decimal places allowed. The 
        second-level keys become the columns of the DataFrame.

        :param json_path: Path to a JSON file.
        :return: A pandas DataFrame representing the JSON data.

        See Also
        --------
        :meth:`jsonCDATA`
        :meth:`nmats`

        Example
        -------
        .. code-block:: python

            piece = Score('./test_files/CloseToYou.mei.xml')
            piece.fromJSON(json_path='./test_files/CloseToYou.json')
        """
        with open(json_path) as json_data:
            data = json.load(json_data)
        df = pd.DataFrame(data).T
        df.index = df.index.astype(str)
        return df

    def insertAudioAnalysis(self, output_filename, json_path, mimetype='', target=''):
        """
        Insert a <performance> element into the MEI score given the analysis data 
        (`json_path`). The original score must be an MEI file. The JSON data will be 
        extracted via the `.nmats()` method. If provided, the `mimetype` and `target` 
        get passed as attributes to the <avFile> element. The performance element 
        will nest the DataFrame data in the <performance> element as a child of 
        <music> and a sibling of <body>. A new file will be saved to the 
        `output_filename` in the output_files directory.

        .. parsed-literal::

            <music>
                <performance xml:id="pyAMPACT-1">
                    <recording xml:id="pyAMPACT-2">
                        <avFile mimetype="audio/aiff" target="song.wav" xml:id="pyAMPACT-3" />
                        <when absolute="00:00:12:428" xml:id="pyAMPACT-4" data="#note_1">
                            <extData xml:id="pyAMPACT-5">
                                <![CDATA[>
                                    {"ppitch":221.30926295063591, "jitter":0.7427361, ...}
                                ]]>
                            </extData>
                        </when>
                        <when absolute="00:00:12:765" xml:id="pyAMPACT-6" data="#note_2">
                        ...
                    </recording>
                </performance>
                <body>
                    ...
                </body>
            </music>

        :param output_filename: The name of the output file.
        :param json_path: Path to a JSON file containing analysis data.
        :param mimetype: Optional MIME type to be set as an attribute to the <avFile> 
            element.
        :param target: Optional target to be set as an attribute to the <avFile> 
            element.
        :return: None but a new file is written

        See Also
        --------
        :meth:`nmats`
        :meth:`toKern`

        Example
        -------
        .. code-block:: python

            piece = Score('./test_files/CloseToYou.mei.xml')
            piece.insertAudioAnalysis(output_filename='newfile.mei.xml'
                json_path='./test_files/CloseToYou.json',
                mimetype='audio/aiff',
                target='Close to You vocals.wav')
        """
        performance = ET.Element('performance', {'xml:id': next(_idGen)})
        recording = ET.SubElement(performance, 'recording', {'xml:id': next(_idGen)})
        avFile = ET.SubElement(recording, 'avFile', {'xml:id': next(_idGen)})
        if mimetype:
            avFile.set('mimetype', mimetype)
        if target:
            avFile.set('target', target)
        nmats = self.nmats(json_path, True)
        # TODO: how do we know which nmat to use when writing the file?
        df = nmats[self.partNames[0]].iloc[:, 7:].dropna(how='all')
        for ndx in df.index:
            when = ET.SubElement(recording, 'when', {'absolute': '00:00:12:428', 'xml:id': next(_idGen), 'data': f'#{ndx}'})
            ET.SubElement(when, 'extData', {'xml:id': next(_idGen), 'text': f'<![CDATA[>{df.loc[ndx].to_dict()}]]>'})
        musicEl = self._meiTree.find('.//music')
        musicEl.insert(0, performance)
        if not output_filename.endswith('.mei.xml'):
            output_filename = output_filename.split('.', 1)[0] + '.mei.xml'
        _indentMEI(self._meiTree)
        # get header/ xml descriptor from original file
        with open(self.path, 'r') as f:
            lines = []
            for line in f:
                if '<mei ' in line:
                    break
                lines.append(line)
        header = ''.join(lines)
        with open(f'./output_files/{output_filename}', 'w') as f:
            f.write(header)
            ET.ElementTree(self._meiTree).write(f, encoding='unicode')

    def show(self, start=None, end=None):
        """
        Print a VerovioHumdrumViewer link to the score in between the `start` and
        `end` measures (inclusive).

        :param start: Optional integer representing the starting measure. If `start` 
            is greater than `end`, they will be swapped.
        :param end: Optional integer representing the ending measure.
        :return: None but a url is printed out

        See Also
        --------
        :meth:`toKern`

        Example
        -------
        .. code-block:: python

            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K398/Krn/K398.krn')
            piece.show(5, 10)
        """
        if isinstance(start, int) and isinstance(end, int) and start > end:
            start, end = end, start
        tk = self.toKern()
        if start and start > 1:
            header = tk[:tk.index('\n=') + 1]
            headerColCount = header.rsplit('\n', 1)[-1].count('\t')
            startIndex = tk.index(f'={start}')
            fromStart = tk[startIndex:]
            fromStartColCount = fromStart.split('\n', 1)[0].count('\t')
            # add the last divisi line to try to get the column count right
            if fromStartColCount > headerColCount:
                divisi = [fromStart]
                firstLines = tk[:startIndex - 1].split('\n')
                for line in reversed(firstLines):
                    if '*^' in line:
                        divisi.append(line)
                        if fromStartColCount - len(divisi) < headerColCount:
                            break
                fromStart = '\n'.join(reversed(divisi))
            tk = header + fromStart
        if end and end + 1 < self._measures().iloc[:, 0].max():
            tk = tk[:tk.index(f'={end + 1}')]
        encoded = base64.b64encode(tk.encode()).decode()
        if len(encoded) > 1900:
            print('''\nWarning: this excerpt is too long to be passed in a url. Instead to see\
            \nthe whole score you can run .toKern("your_file_name"), then drag and drop\
            \nthat file to VHV: https://verovio.humdrum.org/''')
        else:
            print(f'https://verovio.humdrum.org/?t={encoded}')

    def _kernHeader(self):
        """
        Return a string of the kern format header global comments.

        The header includes the composer and title metadata.

        :return: A string representing the kern format header.
        """
        data = [
            f'!!!COM: {self.metadata["composer"]}',
            f'!!!OTL: {self.metadata["title"]}'
        ]
        return '\n'.join(data)

    def _kernFooter(self):
        """
        Return a string of the kern format footer global comments.

        The footer includes the translation date and other relevant metadata.

        :return: A string representing the kern format footer.
        """
        from datetime import datetime
        data = [
            '!!!RDF**kern: %=rational rhythm',
            '!!!RDF**kern: l=long note in original notation',
            '!!!RDF**kern: i=editorial accidental',
            f'!!!ONB: Translated from a {self.fileExtension} file on {datetime.today().strftime("%Y-%m-%d")} via AMPACT',
            '!!!title: @{OTL}'
        ]
        return '\n'.join(data)

    def toKern(self, path_name='', data='', lyrics=True, dynamics=True):
        """
        Create a kern representation of the score. If no `path_name` variable is
        passed, then returns a pandas DataFrame of the kern representation. 
        Otherwise a file is created or overwritten at the `path_name` path. If 
        path_name does not end in '.krn' then this file extension will be added 
        to the path. If `lyrics` is `True` (default) then the lyrics for each part 
        will be added to the output, if there are lyrics. The same applies to 
        `dynamics`.

        :param path_name: Optional string representing the path to save the kern 
            file.
        :param data: Optional string representing the data to be converted to kern 
            format.
        :param lyrics: Boolean, default True. If True, lyrics for each part will 
            be added.
        :param dynamics: Boolean, default True. If True, dynamics for each part 
            will be added.
        :return: String of new kern score if no `path_name` is given, or None if
            writing the new kern file to the location of `path_name`

        See Also
        --------
        :meth:`show`

        Example
        -------
        .. code-block:: python

            # create a kern file from a different symbolic notation file
            piece = Score('https://github.com/alexandermorgan/TAVERN/blob/master/Mozart/K179/Krn/K179.xml')
            piece.toKern()
        """
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
                partNum = self.partNames.index(col) + 1 if col in self.partNames else -1
                firstTokens.append('**kern')
                partNumbers.append(f'*part{partNum}')
                staves.append(f'*staff{partNum}')
                instruments.append('*Ivox')
                partNames.append(f'*I"{col}')
                shortNames.append(f"*I'{col[0]}")
                if includeLyrics and col in lyr.columns:
                    lyrCol = lyr[col]
                    lyrCol.name = 'Text_' + lyrCol.name
                    _cols.append(lyrCol)
                    firstTokens.append('**text')
                    partNumbers.append(f'*part{partNum}')
                    staves.append(f'*staff{partNum}')
                if includeDynamics and col in dyn.columns:
                    dynCol = dyn[col]
                    dynCol.name = 'Dynam_' + dynCol.name
                    _cols.append(dynCol)
                    firstTokens.append('**dynam')
                    partNumbers.append(f'*part{partNum}')
                    staves.append(f'*staff{partNum}')
            events = pd.concat(_cols, axis=1)
            ba = self._barlines()
            ba = ba[ba != 'regular'].dropna().replace({'double': '||', 'final': '=='})
            ba.loc[self.score.highestTime, :] = '=='
            if data:
                cdata = self.fromJSON(data).reset_index(drop=True)
                cdata.index = events.index[:len(cdata)]
                firstTokens.extend([f'**{col}' for col in cdata.columns])
                self._addTieBreakers((events, cdata))
                events = pd.concat((events, cdata), axis=1)
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
                    index=[-12, -11, -10, -9, -8, -7, int(self.score.highestTime + 1)]).fillna('*')
            partTokens.columns = events.columns
            to_concat = [partTokens, de, me, ds, clefs, ks, ts, events, ba]
            if isinstance(events.index, pd.MultiIndex):
                self._addTieBreakers(to_concat)
            body = pd.concat(to_concat).sort_index(kind='mergesort')
            if isinstance(body.index, pd.MultiIndex):
                body = body.droplevel(1)
            body = body.fillna('.')
            for colName in [col for col in body.columns if '__' in col]:
                divStarts = np.where(body.loc[:, colName] == '*^')[0]
                divEnds = np.where(body.loc[:, colName] == '*v')[0]
                colIndex = body.columns.get_loc(colName)
                for _ii, startRow in enumerate(divStarts):
                    if _ii == 0:  # delete everying in target cols up to first divisi
                        body.iloc[:startRow + 1, colIndex] = np.nan
                    else:  # delete everything from the last divisi consolidation to this new divisi
                        body.iloc[divEnds[_ii - 1] + 1: startRow + 1, colIndex] = np.nan
                    if _ii + 1 == len(divStarts) and _ii < len(divEnds):  # delete everything in target cols after final consolidation
                        body.iloc[divEnds[_ii] + 1:, colIndex] = np.nan

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

    def toMEI(self, file_name='', data=''):
        """
        Create an MEI representation of the score. If no `file_name` is passed
        then returns a string of the MEI representation. Otherwise a file called
        `file_name` is created or overwritten in the `output_files` directory.
        If `file_name` does not end in '.mei.xml' or '.mei', then the `.mei.xml`
        file extension will be added to the `file_name`.

        :param file_name: Optional string representing the name to save the new
            MEI file to in the `output_files` directory.
        :param data: Optional string of the path of score data in json format to
            be added to the the new mei file.
        :return: String of new MEI score if no `file_name` is given, or None if
            writing the new MEI file to `output_files/<file_name>.mei.xml`

        See Also
        --------
        :meth:`toKern`

        Example
        -------
        .. code-block:: python

            # create an MEI file from a different symbolic notation file
            piece = Score('kerntest.krn')
            piece.toMEI(file_name='meiFile.mei.xml')
        """
        import pdb
        key = ('toMEI', data)
        if key not in self._analyses:
            root = ET.Element('mei', {'xmlns': 'http://www.music-encoding.org/ns/mei', 'meiversion': '5.0.0'})
            meiHead = ET.SubElement(root, 'meiHead')
            fileDesc = ET.SubElement(meiHead, 'fileDesc')
            titleStmt = ET.SubElement(fileDesc, 'titleStmt')
            title = ET.SubElement(titleStmt, 'title')
            author = ET.SubElement(titleStmt, 'author')
            composer = ET.SubElement(titleStmt, 'composer')
            music = ET.SubElement(root, 'music')
            # insert performance element here
            body = ET.SubElement(music, 'body')
            mdiv = ET.SubElement(body, 'mdiv')
            score = ET.SubElement(mdiv, 'score')
            section = ET.SubElement(score, 'section')

            events = self._parts(compact=True, number=True)
            e0 =events.copy()
            # events.columns = range(1, len(events.columns) + 1 )
            events['Measure'] = self._measures().iloc[:, 0]
            e1 = events.copy()
            # need to assign column names in format (partNumber, voiceNumer) with no splitting up of chords
            events.iloc[:, -1].ffill(inplace=True)
            e2 = events.copy()
            events = events.set_index('Measure')
            e3 = events.copy()
            stack = events.stack((0, 1)).sort_index(level=[0, 1, 2])
            for measure in stack.index.levels[0]:
                meas_el = ET.SubElement(section, 'measure', {'n': f'{measure}'})
                for staff in stack.index.get_level_values(1).unique():   # stack.index.levels[1] doesn't work for some reason
                    if staff == 'Measure':
                        pdb.set_trace()
                    staff_el = ET.SubElement(meas_el, 'staff', {'n': f'{staff}'})
                    for layer in stack.index.levels[2]:
                        layer_el = ET.SubElement(staff_el, 'layer', {'n': f'{layer}'})
                        for nrc in stack.loc[[measure, staff, layer]].values:
                            if nrc.isNote:
                                note_el = ET.SubElement(layer_el, 'note', {'oct': f'{nrc.octave}',
                                        'pname': f'{nrc.step.lower()}', 'xml:id': next(_idGen)})
                                if nrc.duration.isGrace:
                                    note_el.set('grace', 'acc')
                                else:
                                    note_el.set('dur', f'{int(4 / nrc.duration.quarterLength)}')
                                # ET.SubElement(note_el, 'tie', {'type': 'start'})
                                # ET.SubElement(note_el, 'tie', {'type': 'stop'})
                            elif nrc.isRest:
                                rest_el = ET.SubElement(layer_el, 'rest', {'xml:id': next(_idGen)})
                                rest_el.set('dur', f'{int(4 / nrc.duration.quarterLength)}')
                            else:
                                # chord_el = ET.SubElement(layer_el, 'chord')
                                for note in nrc.notes:
                                    chord_note_el = ET.SubElement(layer_el, 'note', {'oct': f'{note.octave}',
                                            'pname': f'{note.step.lower()}', 'xml:id': next(_idGen)})
                                    if note.duration.isGrace:
                                        chord_note_el.set('grace', 'acc')
                                    else:
                                        chord_note_el.set('dur', f'{int(4 / note.duration.quarterLength)}')
            _insertScoreDef(root, self.partNames)
            _indentMEI(root)
            self._analyses[key] = ET.ElementTree(root)

        if not file_name:
            return self._analyses[key]
        else:
            if not (file_name.endswith('.mei.xml') or file_name.endswith('.mei')):
                file_name += '.mei.xml'
            with open(f'./output_files/{file_name}', 'w') as f:
                f.write(_meiDeclaration)
                self._analyses[key].write(f, encoding='unicode')