package org.example;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
//import org.deeplearning4j.nn.multilayer.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.File;
import java.io.IOException;


public class MLPClassifierLinear
{
    public static void main( String[] args ) throws IOException, InterruptedException {
        System.out.println( "Hello World!" );
        int seed=123;      // reproducing output of result by model
        double learningRate=0.02;
        int batchSize=50;
        int nEpochs=30;
        int numInputs=2;
        int numOutputs=2;
        int numHiddenNodes=20;

        //load training data

        RecordReader rr=new CSVRecordReader();
        rr.initialize(new FileSplit(new File("resources/Iris.csv")));
        DataSetIterator tranIter= new RecordReaderDataSetIterator(rr,batchSize,0,2);//(recordreader,batchsize,index of Label column, number of possible labels)//

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Sgd(0.01))
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.valueOf("relu"))
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.valueOf("softmax")).weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        System.out.println(conf.toJson());



    }
}
