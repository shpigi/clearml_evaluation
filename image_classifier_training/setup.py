from setuptools import setup, find_packages


setup(
    name="image_classifier_training",
    version=0.01,
    description="train an image classifier",
    license="Private",
    url="tbd",  # https://github.com/sulfurheron/contrastive_train",
    packages=find_packages(),
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3.8"],
    install_requires=[
        "clearml",
        "jupyter",
        "tensorboard_logger",
        "timm",
        "black[jupyter]",
        "fastai",
        "torch==1.11.0",
        "torchvision==0.12.0",
        "protobuf==3.20.*",
        "tensorboard",
    ],
    # tests_require=[
    #     "pytest",
    # ],
    # setup_requires=[
    #     "pytest-runner",
    # ],
)
