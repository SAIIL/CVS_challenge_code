from setuptools import setup 
  
setup( 
    name='cvs_challenge', 
    version='0.1', 
    description='Sample code for CVS challenge - metrics computation, data loading.', 
    packages=['util','datasets'], 
    install_requires=[ 
        'numpy', 
        'typing', 
        'scikit-learn', 
    ], 
) 
