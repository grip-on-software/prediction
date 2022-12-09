# Prediction

This repository contains models and runners for building and using pattern 
recognition models (based on machine learning and effort estimation) for 
classification and estimation of features that describe Scrum sprints.

The runners and contexts for the classification/estimation models ensure that 
the data set is provided in a certain format, selects features, and splits into 
train/test/validation sets, which allows runs on combinations of features or 
time-travel through the data set (providing different train slices split 
temporally).

The prediction models are based on TensorFlow and Keras. Some models are 
hand-made, based on equations described for those models (such as analogy-based 
effort estimation, ABE) and some are based on existing TF or Keras models (such 
as MLP and DNN).

## Installation

The predictions run using Python 3.7. A number of Python libraries are required 
to run the predictions, which are listed in `requirements.txt`. To install 
these, run `pip3 install tensorflow==$TENSORFLOW_VERSION` and `pip3 install -r 
requirements.txt`, where you replace the `$TENSORFLOW_VERSION` variable with 
a supported [TensorFlow version](#tensorflow-version), and assuming `pip3` is 
a Pip executable for the Python 3 environment (if you do not have Pip, then 
either install it through your package manager or use another method to obtain 
it, see [Pip installation](https://pip.pypa.io/en/stable/installation/)). If 
you do not have permission to install system-wide libraries, then either add 
`--user` after the `install` command, or use a [virtual 
environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments).

The predictions can also be run using Docker. This repository contains 
a `Dockerfile` which can be used to build a Docker image. If you have Docker 
installed, then run `docker build -t gros-prediction .` to build this image. If 
you want to use a different TensorFlow version, then add `--build-arg 
TENSORFLOW_VERSION=$TENSORFLOW_VERSION`, where you replace the 
`$TENSORFLOW_VERSION` variable with an [appropriate 
tag](https://www.tensorflow.org/install/docker#download_a_tensorflow_docker_image) 
of the upstream [tensorflow](https://hub.docker.com/r/tensorflow/tensorflow) 
image for a supported [TensorFlow version](#tensorflow-version). If you add 
`-gpu` at the end, then you can select a GPU device to pin data and models to, 
but in this case you must have 
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker), CUDA toolkit and an 
NVIDIA driver for your GPU installed. Details for setting this up are outside 
the scope of this repository, but some documentation for GPU-enabled Jenkins 
agents may be found elsewhere in the Grip on Software documentation.

To run a Docker image for a prediction, use `docker run -v /path/to/data:/data 
gros-prediction python tensor.py --filename /data/sprint_features.arff 
--results /data/sprint_results.json`. Adjust the `-v` argument to a volume 
specification with some valid paths, perhaps `$PWD`. Add other arguments that 
you would use to configure the runner and model. For a GPU-enabled image, you 
must instead use `docker run --runtime=nvidia ...` and also select an 
appropriate GPU device by adding `--device /gpu:0` at the end. The number in 
this argument selects the GPU to pin to if there are multiple, starting from 
index 0. In case of multiple GPUs, it is also recommended to set the 
environment variable `CUDA_VISIBLE_DEVICES=0` to this index, so that other GPUs 
are not used by TensorFlow to reserve memory. This allows multiple GPUs to be 
used concurrently (for example on a Jenkins agent with multiple executors, or 
just to keep the other GPU available to other users or even a graphical 
desktop).

### TensorFlow version

For both the Docker-based and direct (pip/virtualenv) installation, we have to 
install a specific version of TensorFlow to work with the current models. This 
means that the most recent version will not function. The code has been tested 
with TensorFlow versions 1.12.0 and 1.13.2, but may work with later TensorFlow 
versions before version 2. Note that these versions may get stale by now which 
causes them to not support recent Python versions. This may mean that the PyPI 
registry does not provide packages for pip to install for your Python version. 
This means that we currently require Python 3.7 in order to install TensorFlow 
properly. If this hinders the direct installation because your package manager 
no longer provides Python 3.7, then consider using Docker instead.

## Configuration

The configuration of the models and runners takes place through command line 
arguments provided to the `tensor.py` script. The `--help` argument indicates 
the available options.

When the `--store` argument is set to `owncloud`, then additional configuration 
is loaded from a `config.yml` file (the file path can be changed with the 
`--config` argument). This YAML file must have an object with an `owncloud` key 
where the value is an object with the following keys:

- `url`: The URL to the ownCloud instance from which to load files.
- `verify`: A boolean that indicates whether to verify SSL certificates when 
  connecting to the ownCloud instance.
- `username`: The username to log in as.
- `password`: The password for the username to log in as. If the `keyring` 
  configuration item is true, then this value is ignored.
- `keyring`: Whether to obtain the password from a keyring (e.g. GNOME). The 
  keyring must have a section called `owncloud` and the password is obtained 
  from the username determined by the `username` configuration item.

## Data and running

The prediction runner expects an input dataset to be provided in the form of an 
ARFF file. The first two columns of the ARFF file must be `project_id` and 
`sprint_id`, respectively, and if there is a column named `organization`, then 
it must be the last column and be a nominal attribute of quoted strings.

All other columns (attributes) are considered to be numeric. Each row 
(instance) in the data set describes features of a single sprint. Except for 
the three metadata attribute fields, the attributes can have missing values 
(question marks) and may be selected by the runner.

A time attribute, if provided and indicated by the `--time` argument, may be 
used to split the dataset temporally for time-travel, and should be a number 
based on a sprint's start date at a high enough resolution (e.g. the number of 
days since an origin) to make a realistic split (further binning to reduce the 
number of splits, or combine them, is controlled by the `--time-size` and 
`--time-bin` arguments).

Selection by the runner is controlled by the `--index`, `--remove` and 
`--label` arguments. Like `--time`, these should indicate indexes to add, 
remove or consider as a label to predict. For `--index` and `--remove`, 
multiple indexes can be provided using commas or spaces as separation. Indexes 
may be either a positive number (starting at 0 until the number of columns, 
exclusive), a negative number (from -1 for the last column until the additive 
inverse of the number of columns, inclusive), or the name of the column.

Additional column can be generated using `--assign`, where the argument must be 
a Python assignment expression. The expression can refer to other names of 
columns as variables, and a limited number of functions is available. Note that 
it may be preferable to perform this calculation beforehand, which may allow 
tracking more metadata outside of this repository. Note that the `--label` 
argument may also be an expression, but without an assignment.

The ARFF file can be provided by the `data-analysis` repository, using the 
`features.r` script to collect, analyze and output features. This repository 
contains a `Jenkinsfile` with appropriate steps for a Jenkins CI deployment, 
with example arguments to the scripts.

After a prediction has finished, the `tensor.py` runner outputs a JSON file to 
`sprint_results.json` (or another path determined by the `--results` argument) 
which contains predictions for a validation set, model metrics, metadata and 
configuration. This format can be read the `sprint_results.r` script from the 
`data-analysis` repository in order to combine the prediction model results 
with other sprint data so that it can be used in an API for the 
`prediction-site` visualization.
