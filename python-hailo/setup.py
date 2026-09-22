from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="trustmark-hailo",
    version="0.1.0",
    python_requires=">=3.9",
    description="TrustMark encoder/decoder running on the Hailo-8L NPU (Raspberry Pi 5 AI Kit)",
    url="https://github.com/adobe/trustmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="John Collomosse",
    author_email="collomos@adobe.com",
    license="MIT",
    packages=find_packages(include=["trustmark_hailo*"]),
    # secret2image_{TYPE}.npz ship in trustmark_hailo/models/ rather than being
    # downloaded on first use - box_head_{TYPE}.npz is NOT included (~55MB,
    # too big to ship; downloaded on first use instead, same as the .hef files).
    package_data={"trustmark_hailo": ["models/secret2image_*.npz"]},
    include_package_data=True,
    # not that numpy 2.x breaks hailo_platform
    install_requires=["numpy<2", "pillow"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
