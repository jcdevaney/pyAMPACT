# ![AMPACT](https://static.tumblr.com/3675cda2cf7fe706ff09e8dbb590f657/ytsja31/iXwor8hkf/tumblr_static_10unxb8apbdw4k8owooo0kg8s.jpg)

AMPACT is a project run by [Prof. Johanna Devaney](https://www.brooklyn.cuny.edu/web/academics/schools/mediaarts/faculty_details.php?faculty=1368) of Brooklyn College. You can find out more about the project and read related papers and documentation on the [AMPACT Tumblr page](https://ampact.tumblr.com). One of primary goals of this project is to make an practical and automated connection between corresponding scores in symbolic notation, audio recordings of performances, and audio analysis.

This code repository translates some of the existing tooling written in matlab to python, primarily using the [pandas](https://pandas.pydata.org) and [music21](https://web.mit.edu/music21/) libraries and draws many lessons from the [crim-intervals](https://github.com/HCDigitalScholarship/intervals), [humlib](https://github.com/craigsapp/humlib), and [vis-framework](https://github.com/ELVIS-Project/vis-framework) repositories.

**Most significantly**, this repo makes a `Score` object available. You can use this `Score` object to get tables of various types of information or analysis about a piece. Explore the primary features of this repository in the README.ipynb on github, or live with this [colab notebook](https://githubtocolab.com/alexandermorgan/AMPACT/blob/main/README.ipynb).

# Directory Information
1. As in the original MATLAB application, functions are currently separated by file, with the entry point being the `exampleScript.py` file, calling appropriate layered functionality within the directory.
2. The `./sym_functions` directory houses the functionality of the symbolic and MIDI processing, housed in the `script.py` file, and called from the parent directory as needed.
3. The `./unit_tests` directory houses choice testing scripts for functions throughout the application, with the `./unit_test/in_progress` directory housing in-flight tests, tests to be used later, or tests that were once used on deprecated functions.
4. Similarly, the `./unused_scripts` directory houses functions to be used later, or sometimes ones that have been deprecated or refactored.
5. Finally, `./placeholders` houses any hardcoded files necessary for filling in gaps of functionality as the application is developed.


# Bug notes:
11/13:
- runAlignment issues:
  - row 2 of the select_state array needs calculation, this is the cumsumvals2 var // rows 1 and 3 are good:
    - row 2 is calculated by the selectStates function, but there are issues with reshaping
- runHMMAlignment issues:
  - obs is not populated because of the lack of yinres functionality
    - this effects the calculation of like and prior and pLike variables
    - these values are hardcoded for now
  - the vpath and state_ord vars are the only ones needed from the runHMM method
    - vpath effected by the above
    - state_ord is good

Hardcoded pieces, taken from MATLAB:
# - 2nd row of selectState var in runAlignment
# - prLike vars in runHMMAlignment