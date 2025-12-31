from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirement_list=[]
    try:
        with open('requirements.txt', 'r') as file:
            requirements = file.readlines()
            for line in requirements:
                r=line.strip()
                if r and r!='-e .':
                    requirement_list.append(r)

    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirement_list

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Aaditya Deshpande",
    author_email="desh.aaditya165@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)