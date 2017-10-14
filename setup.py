from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
from setuptools import setup, find_packages


HERE = os.path.dirname(os.path.abspath(__file__))

# import ``__version__`` from code base
exec(open(os.path.join(HERE, 'sparsenet', 'version.py')).read())


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


setup(name='sparsenet',
      version=__version__,
      description='A library for Structured Sparsity in Python.',
      author='Joshua D. Loyal',
      author_email='joshua.d.loyal@gmail.com',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extra_require={
          'notebooks': ['jupyter>=1.0.0']
      },
      test_require=['pytest', 'pytest-pep8'],
      keywords=['machine-learning', 'statistics'],
      )
