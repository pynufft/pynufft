rm -r dist build pynufft.egg-info
python setup.py sdist bdist_wheel
#python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
#pip install --index-url https://test.pypi.org/simple/ --no-deps example-pkg-YOUR-USERNAME-HERE
