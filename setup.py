from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


with open("requirements-dev.txt") as f:
    tests_require = [
        line for line in f.read().splitlines() if line != "-r requirements.txt"
    ]

setup(
    name="lib",
    version=0.01,
    description="supreme_model",
    author="Kesimoglu, Ziynet Nesibe",
    author_email="ziynetnesibe.kesimoglu@nih.gov",
    license="",
    url="",
    scripts=[],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    namespace_packages=[],
    py_modules=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={},
    data_files=[],
    package_data={},
    install_requires=install_requires,
    dependency_links=[],
    zip_safe=True,
    keywords="",
    python_requires=">=3",
    obsoletes=[],
)
