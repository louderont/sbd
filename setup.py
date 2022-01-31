from setuptools import setup

setup(
    name='sbdetection',
    version='0.0.1',
    py_modules=['sbdetection'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'sbdetection = sbdetection.main:run',
        ],
    },
)