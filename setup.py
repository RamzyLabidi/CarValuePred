from setuptools import find_packages, setup

from typing import List

EHYPHEN = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]
        if EHYPHEN in requirements:
            requirements.remove(EHYPHEN)
    return requirements

setup(
    name = "CarPredproject",
    version = "0.0.1",
    author = "Ramzy",
    author_email = "ramzylabidi43@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)