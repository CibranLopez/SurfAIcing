from distutils.core import setup

with open('LICENSE') as f:
    license = f.read()

setup(
  name = 'SurfAIcing',
  packages = ['SurfAIcing'],
  version = '1.0',
  license=license,
  description = 'Automated analysis of surface energetics.',
  long_description_content_type='docs/index.rst',
  author = 'Cibrán López Álvarez',
  author_email = 'cibran.lopez@upc.edu',
  url = 'https://github.com/IonRepo/IonDiff',
  download_url = 'https://github.com/IonRepo/IonDiff/archive/refs/tags/0.1.tar.gz',
  keywords = ['Energy Materials', 'Surface Energetics', 'Machine Learning'],
  install_requires=[
          'numpy',
          'matgl',
          'ase',
          'mace-torch',
          'seaborn',
          'pymatgen',
          'openpyxl',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
