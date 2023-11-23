<h1 align="center">pyAMPACT</h1>

<p align="center">
    <img alt='AMPACT' src="https://static.tumblr.com/3675cda2cf7fe706ff09e8dbb590f657/ytsja31/iXwor8hkf/tumblr_static_10unxb8apbdw4k8owooo0kg8s.jpg"/>
</p>

pyAMPACT is a project run by [Prof. Johanna Devaney](https://www.brooklyn.cuny.edu/web/academics/schools/mediaarts/faculty_details.php?faculty=1368) of Brooklyn College. You can find out more about the project and read related papers and documentation on the [AMPACT Tumblr page](https://ampact.tumblr.com). One of primary goals of this project is to make an practical and automated connection between corresponding scores in symbolic notation, audio recordings of performances, and audio analysis.

This code repository translates some of the existing tooling written in matlab to python, primarily using the [pandas](https://pandas.pydata.org) and [music21](https://web.mit.edu/music21/) libraries and draws many lessons from the [crim-intervals](https://github.com/HCDigitalScholarship/intervals), [humlib](https://github.com/craigsapp/humlib), and [vis-framework](https://github.com/ELVIS-Project/vis-framework) repositories.

**Most significantly**, this repo makes a `Score` object available. You can use this `Score` object to get tables of various types of information or analysis about a piece. Explore the primary features of this repository in the README.ipynb on github, or live with this [colab notebook](https://githubtocolab.com/alexandermorgan/AMPACT/blob/main/README.ipynb).

