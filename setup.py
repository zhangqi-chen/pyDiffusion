from setuptools import setup

setup(
    name='pydiffusion',
    version='0.2',
    packages=['pydiffusion'],
    include_package_data=True,
    python_requires='>=3',
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'],

    # metadata
    author='Zhangqi Chen',
    author_email='wshchzhq@gmail.com',
    description='A Python library for diffusion simulation and data analysis',
    license='MIT',
    url='https://github.com/zhangqi-chen/pyDiffusion',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
)
