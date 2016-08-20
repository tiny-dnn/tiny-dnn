# tiny-dnn documentations

A built version of document is available at [http://tiny-dnn.readthedocs.io/en/latest/index.html](here).

## Local build

You can build html documents in your local machine if you prefer.
Assuming you have python already, install Sphinx and recommonmark at first:

```bash
$ pip install sphinx sphinx-autobuild
$ pip install recommonmark
```

#### build on Windows
```bach
cd docs
make.bat html
```

#### build on Linux
```bash
cd docs
make html
```