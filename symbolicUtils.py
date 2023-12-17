import json
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET

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
function_pattern = re.compile('[^TtPpDd]')
imported_scores = {}
tinyNotation_pattern = re.compile("^[-0-9a-zA-Zn _/'#:~.{}=]+$")
volpiano_pattern = re.compile(r'^\d--[a-zA-Z0-9\-\)\?]*$')

meiDeclaration = """<?xml version="1.0" encoding="UTF-8"?>
<!-- <?xml-model href="../../Documents/music-encoding/dist/schemata/mei-all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>-->
<?xml-model href="https://music-encoding.org/schema/dev/mei-all.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>
"""

reused_docstring =  """
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
-------
.. code-block:: python

    piece = Score('https://raw.githubusercontent.com/alexandermorgan/TAVERN/master/Mozart/K455/Stripped/M455_00_03c_a.krn')
    piece.harm()
"""


def addTieBreakers(partList):
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

def clefHelper(clef):
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

def combineRests(col):
    """
    Helper function for the `notes` method. 

    Combine consecutive rests in a given voice. Non-first consecutive rests 
    will be removed.

    :param col: A pandas Series representing a voice.
    :return: The same pandas Series with consecutive rests combined.
    """
    col = col.dropna()
    return col[(col != 'r') | ((col == 'r') & (col.shift(1) != 'r'))]

def combineUnisons(col):
    """
    Helper function for the `notes` method. 

    Combine consecutive unisons in a given voice. Non-first consecutive unisons 
    will be removed.

    :param col: A pandas Series representing a voice.
    :return: The same pandas Series with consecutive unisons combined.
    """
    col = col.dropna()
    return col[(col == 'r') | (col != col.shift(1))]

def fromJSON(json_path):
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

        newID = next(idGen)
    """
    while True:
        yield f'pyAMPACT-{start}'
        start += 1
idGen = _id_gen()

def indentMEI(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indentMEI(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def insertScoreDef(root, part_names=[]):
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
        scoreDef = ET.Element('scoreDef', {'xml:id': next(idGen), 'n': '1'})
        if len(part_names) == 0:
            part_names = sorted({f'Part-{staff.attrib.get("n")}' for staff in root.iter('staff')})
        for i, staff in enumerate(part_names):
            staffDef = ET.SubElement(scoreDef, 'staffDef', {'label': staff, 'n': str(i + 1), 'xml:id': next(idGen)})
            ET.SubElement(staffDef, 'label', {'text': staff, 'xml:id': next(idGen)})
        scoreEl = root.find('.//score')
        if scoreEl is not None:
            scoreEl.insert(0, scoreDef)

def _kernChordHelper(_chord):
    """
    Parse a music21 chord object into a kern chord token.

    This method uses the `_kernNoteHelper` method to convert each note in the 
    chord into a kern note token. The tokens are then joined together with 
    spaces to form the kern chord token.

    :param _chord: A music21 chord object to be converted into a kern chord token.
    :return: A string representing the kern chord token.
    """
    return ' '.join([_kernNoteHelper(note) for note in _chord.notes])

def kernFooter(fileExtension):
    """
    Return a string of the kern format footer global comments.

    The footer includes the translation date and other relevant metadata.

    :return: A string representing the kern format footer.
    """
    from datetime import datetime
    return f"""!!!RDF**kern: %=rational rhythm
        !!!RDF**kern: l=long note in original notation
        !!!RDF**kern: i=editorial accidental
        !!!ONB: Translated from a {fileExtension} file on {datetime.today().strftime("%Y-%m-%d")} via AMPACT
        !!!title: @{{OTL}}"""

def kernHeader(metadata):
    """
    Return a string of the kern format header global comments.

    The header includes the composer and title metadata.

    :return: A string representing the kern format header.
    """
    return f'!!!COM: {metadata["composer"]}\n!!!OTL: {metadata["title"]}'

def _kernNoteHelper(_note):
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

def kernNRCHelper(nrc):
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
        return _kernNoteHelper(nrc)
    elif nrc.isRest:
        return f'{_duration2Kern.get(round(float(nrc.quarterLength), 5))}r'
    else:
        return _kernChordHelper(nrc)

def noteRestHelper(nr):
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

def remove_namespaces(doc):
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

def removeTied(noteOrRest):
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

def snapTo(data, snap_to=None, filler='forward', output='array'):
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