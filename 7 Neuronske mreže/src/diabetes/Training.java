/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetes;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.MultiLayerPerceptron;

/**
 *
 * @author Milos Dragovic
 */
public class Training {

    private NeuralNetwork neuralNetwork;
    private double accuracy;

    public Training() {
    }

    public Training(NeuralNetwork neuralNetwork, double accuracy) {
        this.neuralNetwork = neuralNetwork;
        this.accuracy = accuracy;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }
    
    
}
