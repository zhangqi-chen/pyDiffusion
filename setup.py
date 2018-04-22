from setuptools import setup

setup(
    name='pydiffusion',
    version='0.1',
    packages=['pydiffusion'],
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'],

    # metadata
    author='Zhangqi Chen',
    author_email='wshchzhq@gmail.com',
    description='A Python library for diffusion simulation and data analysis',
    license='MIT',
    url='https://github.com/zhangqi-chen/pyDiffusion'
)
