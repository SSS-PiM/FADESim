# FADESim
FADESim: Enable <u>F</u>ast and <u>A</u>ccurate <u>D</u>esign <u>E</u>xploration for Memristive Accelerators (MAs) considering non-idealities.
see https://doi.org/10.1109/TCAD.2024.3485589 for details.

The term FADE in FADESIM has a dual meaning. It refers to non-idealities that decrease computational accuracy and reliability, causing MA
to fade (i.e., lose its functionality). Therefore, FADESIM denotes realistic MA's simulations that consider non-idealities.

The non-idealities supported include **IR-drop, IV non-linearity, c2c varation, noises, and etc..**

If you find this work useful, please cite:

Wu, Bing, Yibo Liu, Jinpeng Liu, Huan Cheng, Xueliang Wei, Wei Tong, and Dan Feng. "FADESIM: Enable Fast and Accurate Design Exploration for Memristive Accelerators Considering Nonidealities," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 44, no. 4, pp. 1529-1543, April 2025.

Bibtex is:
```
@article{wu2024fadesim,
  author={Wu, Bing and Liu, Yibo and Liu, Jinpeng and Cheng, Huan and Wei, Xueliang and Tong, Wei and Feng, Dan},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={FADESIM: Enable Fast and Accurate Design Exploration for Memristive Accelerators Considering Nonidealities}, 
  year={2025},
  volume={44},
  number={4},
  pages={1529-1543},
  publisher={IEEE}
 }
```



Our code consists of two parts:
1. crossbarHspice, for single crossbar array simulation (by SPICE, our fast simulation method, or other fast simluation method).
2. Pimtorch, for algorithm-level simulation using pytorch, which intergates our array-level fast simluation method. 

## CrossbarHspice
This is used to generate spice code for single crossbar simulation.
Our Fast simluation method and some other fast simulation method are also included.

### Usage
  - ```cd crossbarHspice && mkdir build && cd build```
  - ```cmake .. && make -j```
  - A excutable file named "sim" will be created.
  - ```./sim ${conf_file}```

### Configuration File Format
You should configure your file like the following format.
This is an example for 128x128 array simulation with input all 1 to the wordline, and sense the output current on all bitline. Cell is 2 bits with resistance 2e6 1.6e6 1.3e6 1e6.
```
topString .title twoDArray
useRtypeReRAMCell yes
cellRstates 4 2e6 1.6e6 1.3e6 1e6
bottomString .end

topString .OPTIONS  numdgt=10

arraySize 128 128
selector no
line_resistance 2.93

// set all cell to state_2 (1.3e6)
setCellR -1 -1 2

// set wordline (left) are used
setUseLine left -1 
// set bitline (down) are used
setUseLine down -1

// all left connect to 1V
setLineV left -1 1

// all down connect to 0V (ground)
setLineV down -1 0

senseBitlineI down -1

// spice code generated to spice.out
build spice.out

// start our fast simulation method
nodebasedGSMethod 100000 1.95 0 yes 1e-8
```

The results of Fast simulation method will print directly.
SPICE code will generated into spice.out.
Then you need to use hspice or ngspice to simulate the spice.out like ```hspice spice.out``` or ```ngspice -b spice.out```.

More examples and descriptions about configuration format: see our example config in the dir crossbarHspice/examples.

In crossbarHpsice/tests, there are some scripts to help you to do fast comparison for different simulation methods. You should use them after you have known how to do the basic simulations.

### CrossbarHspice version 2
We update the code to V2 in the folder "crossbarHspice/V2". Compared to V1, V2 has a clearer design, making it easier to add new commands in the future.
Now all support for SPICE simulation has been implemented in V2 as similar in V1.
But for fast simulation methods, only our own GS method (Au algorithm in our paper) has been implemented. Other fast simulation methods should use V1 version.

#### Usage
  - ```cd crossbarHspice/V2 && mkdir build && cd build```
  - ```cmake .. && make -j```
  - A excutable file named "sim" will be created.
  - ```./sim ${conf_file}```


## PimTorch
This is used to do the algorithm-level simulation.

### Usage
  1. pytorch, scipy, and etc.. should be installed (conda recommanded). If you see you lack some libs when running, just conda install them. 
  2. ```cd pimtorch && python3 mnist_ir_drop.py --train // train is done without any non-idealities, just to get the model weight. An inference with non-idealities is followed at last.```
  3. ```python3 mnist_ir_drop.py // inference is done with non-idealities```

All our configurations about cell/array/simulation method/weight and etc., are in pimtorch/pimfixedpoint/fixedPoint/nn/commonConst.py.


 
