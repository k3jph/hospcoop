import hospcoop

from setuptools import setup

setup(name = hospcoop.__name__,
      version = hospcoop.__version__,
      description = 'Hospital Cooperation During Acute Resource Shortages',
      author = '"Anna Camille Svirsko" <anna.svirsko@usna.edu>, "James P. Howard, II" <james.howard@jhu.edu>"',
      license = 'MIT',
      packages = ['hospcoop'],
      package_data = {
          'hospcoop' : ["data/*.toml"]
      },
      entry_points = {
          'console_scripts': ['dailysim=hospcoop.dailysim:main'],
      },
      zip_safe = False)