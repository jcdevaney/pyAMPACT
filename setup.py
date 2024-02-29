from setuptools import setup

setup(
    name='pyampact',
    version='0.4.0',    
    description='A example Python package',
    url='https://www.ampact.org',
    author='Johanna Devaney',
    author_email='test@email.com',    
    license='BSD 2-clause',
    packages=['pyampact'],
    install_requires=['music21==9.1.0',
                      'pandas==2.2.0',
                      'pyarrow==15.0.0',
                      'numpy==1.24.3',
                      'requests==2.31.0',
                      'pytest==7.4.3',
                      'scipy==1.11.1',
                      'librosa==0.10.0.post2',
                      'setuptools>=48',                      
                      ],

    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
)    