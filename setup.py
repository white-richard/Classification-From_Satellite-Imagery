from setuptools import setup, find_packages

setup(
    name="classification_from_satellite_imagery",
    version="0.1",
    description="Classifying aircraft from satellite imagery.",
    author="Rich White",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "pillow",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "torchmetrics",
        "scikit-learn",
        "albumentations",
        "pycocotools",
    ],
    entry_points={
        "console_scripts": [
            "train_model=main:train_model",
            "evaluate_model=main:evaluate_model",
            "run_inference=main:run_inference", 
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",

)
