"""setup for xtz"""
from setuptools import setup

setup(name='xtz',
      version='0.0.1',
      description='xtz helps with building linear pipelines',
      url='https://github.com/kdeenanauth/xtz',
      author='Kevin Deenanauth',
      author_email='kevin@deenanauth.com',
      license='MIT',
      packages=['xtz'],
      install_requires=[
          'ptpython',
          'decorator'
          ],
      zip_safe=False)
