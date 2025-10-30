from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:

    """
    This function will return the list of requirements
    
    """
    requirements_lst: List[str] = []
    try:

        with open(file_path, 'r') as file:

            # Read all lines from the file
            lines = file.readlines()

            #Process each line to extract requirements
            for line in lines:
                requirement = line.strip()
                # ignore te empty lines and -e .

                if requirement and not requirement.startswith('-e '):
                    requirements_lst.append(requirement)
    
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")

    return requirements_lst

print(get_requirements("requirements.txt"))



setup(
    name="NetworkSecurity",
    version="0.1",
    author="Alvin Leonald Kabwama",
    author_email="alvin.kabwama10@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)