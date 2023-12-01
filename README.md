# Extracting crops from whole slide images

Create and acitvate environment using mamba:\
`mamba env create --file scampi_preprocessing/environment.yml --name myenv`\
`conda activate myenv`\

The folder contains scripts for extracting crops to either a HDF5 format or a tfrecords format.

Example of usage where we create a tfrecords file of crops from a set of .mrxs slides:\
`python prepare_tf_records_from_slides --path_to_slides path/to/slides --path_to_tfrecords path/to/destination`
