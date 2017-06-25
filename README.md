**Hierarchial LSTM for Topic Model**
We attempt to build a hierarchial LSTM that will predict document level topics. <br />
The network is composed of three sub-networks: <br />
1) Word level topic model <br />
2) Sentence level topic model <br />
3) Document level topic model <br />

Training is performed in a pipeline. We first train the word level model, then then sentence level model, and finally the document level model. <br />
Inference is performed by chaining these models and outputting a distribution of all possible topics. <br /> 

** Completed **
Vocab builder
Basic data reader
Hierarchial RNN Topic Model
LSTM Topic Model
Train loop for each sub-network


** To Do **
Unit Test HTM, reader, main modules
Write training loop for entire HTM
Write validation loop for HTM subnetworks
Create training, val, and testing data
Train Network
Test Network

