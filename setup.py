from setuptools import setup, find_packages

setup(
    name='astro-refine',
    version='0.1.0',    
    description='Astro Refine',
    url='https://github.com/xinyuesheng/astro-refine',
    author='Xinyue Sheng',
    author_email='xsheng03@qub.ac.uk',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',                     
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)