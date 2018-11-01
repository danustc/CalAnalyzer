'''
This is Dan's first setup.py test! Hopefully it would work.
'''

from setuptools import setup, find_packages

setup(name = 'CalAnalyzer',
        version = '0.01',
        description = 'A small downstream analysis package',
        url = 'https://github.com/danustc/CalAnalyzer',
        author = 'danustc',
        author_email = 'Dan.Xie@ucsf.edu',
        license = 'UCSF',
        packages = find_packages(),
        zip_safe = False
        )
