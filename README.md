# BerryPortraits 
Cranberry image analysis tool described by Loarca et al. (in publication)

This image analysis pipeline was written by Breeding Insight, a USDA-funded program based at Cornell University, in conjunction with the Zalapa Lab based in Madison, WI.

The tool is designed to be run as a standalone python script with no installation. A more user-friendly GUI tool for running this script is actively being worked on by Breeding Insight, for potential release in 2025.

## SETUP
This code is designed to run from within a [Conda](https://anaconda.org/anaconda/conda) environment. We strongly recommend using [mamba](https://mamba.readthedocs.io/), a much faster implementation of the command-line `conda` tool. All `mamba` commands have the same structure and arguments as the relevant `conda` commands.

To install and set up `mamba`:
1. Download the appropriate installer and complete the installation process for your system
2. Open the terminal and confirm that mamba is installed with `mamba -h`
3. Find the correct YAML file for your OS
4. Create a new mamba environment from the appropriate YAML file:
   
`mamba env create -f yml/conda_env_bpt_mac.yml`

*Note:* You may run into dependency issues using the YAML file for your specific platform. If you encounter errors, please try using the `conda_env_bpt_generic.yml` file. Following this, you will need to activate the environment and use `pip` to install an additional library that is not available through `conda` channels:

```
mamba env create -f yml/conda_env_bpt_generic.yml
mamba activate bpt
pip install colorcorrectionML
```

If you still encounter errors, email Tyr Wiesner-Hanks ([tw372@cornell.edu](mailto:tw372@cornell.edu)) for support.


5. Try activating the environment:

`mamba activate bpt`

6. Run the script using the instructions below. When finished, you can simply close out of the terminal or deactivate the environment:

`mamba deactivate`

## RUNNING THE PIPELINE
Each time you run the pipeline, you will need to activate the `mamba` environment from your command-line environment:
`mamba activate bpt`

Once the environment is activated, you are ready to run the script. To list the potential commands:

`python berryportraits.py -h`

The only required argument is an input directory or image: If specifying a single image, it must of format .jpg, .jpeg, .tif, .tiff, or .png (case insensitive). If specifying a directory, the directory must contain at least one image of this format.


`python ./berryportraits.py -i test_dir/`

`python ./berryportraits.py -i test_image.png`

By default, the script will automatically name the output file based on the image or directory name. To change this, use the `-o` flag:

`python ./berryportraits.py -i test_dir -o results.csv`

In most cases, you should have a circular size marker visible in the image. To convert the output into cm, specify the diameter (in cm) with the `-s` flag:

`python ./berryportraits.py -i test_dir -o results_in_cm.csv -s 2.5`

Annotated images can be useful for validating that the pipeline is recognizing and segmenting objects correctly, and for troubleshooting if and when you encounter errors. To save out annotated images, either use the `-w` flag, which will automatically create a new directory based on the input name, or use the `-a` flag to name your own directory:

`python ./berryportraits.py -i test_dir -w   # will create a dir called test_dir_annotated`
`python ./berryportraits.py -i test_dir -a more_annotations`

By default, the script only spawns a single process. To spawn more processes, use the `-n` flag. The number of processes cannot exceed your machine's CPU count as given by `multiprocessing.cpu_count()`:

`python ./berryportraits.py -i test_dir -n 4`

## QUESTIONS/COMMENTS  
Please address all questions to Tyr Wiesner-Hanks ([tw372@cornell.edu](mailto:tw372@cornell.edu))

