/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package breastcancertrain;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author Milos Dragovic
 */
public class Training {

    private NeuralNetwork neuralNet; 
    private double error;

    public Training() {
    }

    public Training(NeuralNetwork neuralNet, double accuracy) {
        this.neuralNet = neuralNet;
        this.error = accuracy;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }
    
    
    
}
