# arctopus/setup.py
from setuptools import setup, find_packages

setup(
    name='arctopus_trainer',
    version='0.1',
    #package_dir={'': 'src'},
    #packages=find_packages(where='src'), # This will find your 'src' package
    

    # This explicitly tells setuptools that the Python package 'arctopus_trainer'
    # corresponds to the 'src' directory in your project structure.
    # This is critical for correct egg-info placement and package discovery.
    package_dir={'arctopus_trainer': 'src'},

    # Explicitly list ONLY the packages required for the training run.
    # - 'arctopus_trainer': This is the top-level package, mapped to your 'src' directory.
    #   (This includes src/common.py as arctopus_trainer.common, as common.py is a module directly under src)
    # - 'arctopus_trainer.training': This includes the 'src/training' subpackage
    #   where your train_codegemma.py resides.
    packages=[
        'arctopus_trainer',
        'arctopus_trainer.training'
    ],


    install_requires=[ # These should largely mirror your requirements.txt
        'torch',
        'transformers',
        'numpy',
        'bitsandbytes', # Needed for 4-bit quantization
        'accelerate',   # Needed for 4-bit quantization and distributed training
        'pandas',       # Potentially useful for data handling
        'python-json-logger',
        'datasets',
        'gcsfs'
        # Add any other libraries your common.py or training code uses
    ],
    python_requires='>=3.9', # Specify Python version compatibility
)