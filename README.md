# FADESim
FADESim: Fast and Accurate Design Exploration for realistic Memristive Accelerators (MAs) considering non-idealities.
see https://doi.org/10.1109/TCAD.2024.3485589 for details.

The term FADE in FADESIM has a dual meaning. It refers to non-idealities that decrease computational accuracy and reliability, causing MA
to fade (i.e., lose its functionality). Therefore, FADESIM denotes simulations that consider non-idealities.

Our code consists of two parts:
1. crossbarHspice, for single crossbar array simulation (by SPICE, our fast simulation method, or other fast simluation method).
2. Pimtorch, for algorithm-level simulation using pytorch, which intergates our array-level fast simluation method. 

## CrossbarHspice
This is used to generate spice code for single crossbar simulation.
Our Fast simluation method and some other fast simulation method are also included.

### Usage
  - cd crossbarHspice && mkdir build && cd build 
  - cmake ..
  - A excutable file named "sim" will be created.
  - ./sim ${conf_file}

### Configuration File Format
You should configure your file like the following format.


More examples about configuration format: see our example config in the dir crossbarHspice/examples.


## PimTorch
This is used to do the algorithm-level simuation.
Being prepared. Wait fews day.

### Usage




 
