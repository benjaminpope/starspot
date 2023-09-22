from setuptools import setup

setup(name='starspot',
      version='0.2',
      description='Tools for measuring stellar rotation periods',
      url='http://github.com/benjaminpope/starspot',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['starspot'],
      install_requires=['numpy', 'pandas', 'h5py', 'tqdm', 'emcee', 'jax', 'numpyro', 'tinygp',
                         'astropy', 'matplotlib', 'scipy','chainconsumer'],
      zip_safe=False)
