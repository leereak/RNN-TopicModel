**Hierarchial LSTM for Topic Model**
We attempt to build a hierarchial LSTM that will predict document level topics. <br />
The network is composed of three sub-networks: <br />
1) Word level topic model <br />
2) Sentence level topic model <br />
3) Document level topic model <br />

Training is performed in a pipeline. We first train the word level model, then then sentence level model, and finally the document level model. <br />
Inference is performed by chaining these models and outputting a distribution of all possible topics. <br /> 

**Completed** <br />
Vocab builder <br />
Basic data reader <br />
Hierarchial RNN Topic Model<br />
LSTM Topic Model <br />
Train loop for each sub-network <br /> <br /> 


**To Do** <br />
Unit Test HTM, reader, main modules<br />
Write training loop for entire HTM <br />
Write validation loop for HTM subnetworks<br />
Create training, val, and testing data <br />
Train Network <br />
Test Network <br />

