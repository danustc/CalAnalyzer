'''
This is Dan's first setup.py test! Hopefully it would work.
'''

from setuptools import setup

setup(name = 'CalAnalysis',
        version = '0.01',
        description = 'A small downstream analysis package',
        url = 'https://github.com/danustc/CalAnalysis',
        author = 'danustc',
        author_email = 'Dan.Xie@ucsf.edu',
        license = 'UCSF',
        packages = ['src'],
        zip_safe = False
        )
