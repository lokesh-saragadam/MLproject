from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this fnction will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements        
setup(

name='MLproject',
version='0.0.1',
author='lokesh',
author_email='lokeshs2k6@gmail.com',
packages=find_packages(),
intsall_requirements=get_requirements('requirements.txt')

)