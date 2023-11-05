from setuptools import setup

setup(name='fastmatch',
      version='0.1',
      description='Fast matching estimators for causal inference',
      url='http://github.com/apoorvalal/fastmatch',
      author='Apoorva Lal',
      author_email='lal.apoorva@gmail.com',
      license='MIT',
      install_requires=[
          'numpy', 'faiss-cpu', 'scikit-learn'
      ],
      packages=['fastmatch'],
      zip_safe=False)
