/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetes;

import com.sun.javafx.font.Metrics;
import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Milos Dragovic
 */
public class DiabetesTrain implements LearningEventListener {

    int inputCount = 8, outputConunt = 1;
    double[] learningRate = {0.2, 0.4, 0.6};
    List<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        new DiabetesTrain().run();
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation mbp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iteration: " + mbp.getCurrentIteration() + " Total network error: " + mbp.getTotalNetworkError());
    }

    private void run() {
        String dataSetFile = "diabetes_data.csv";
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputCount, outputConunt, ",");
        Normalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();
        DataSet[] trainAndTestSets = dataSet.split(0.7, 0.3);
        DataSet trainSet = trainAndTestSets[0];
        DataSet testSet = trainAndTestSets[1];

        int numOfIteration = 0, numOfTraining = 0;
        for (int i = 0; i < learningRate.length; i++) {

            MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(inputCount, 20, 10, outputConunt);
            BackPropagation learningRule = neuralNetwork.getLearningRule();
            learningRule.addListener(this);
            learningRule.setLearningRate(i);
            learningRule.setMaxError(0.07);
            neuralNetwork.learn(trainSet);
            numOfTraining++;
            numOfIteration += learningRule.getCurrentIteration();

            double accuracy = evaluateAcc(neuralNetwork, testSet);
            Training t = new Training(neuralNetwork, accuracy);
            trainings.add(t);
        }
        System.out.println("Srednja vrednost broja iteracija:" + (double) numOfIteration / numOfTraining);
        saveNetWithMaxAccuracy();
    }

    private double evaluateAcc(MultiLayerPerceptron neuralNet, DataSet testSet) {
        MatricaKonfuzije cm = new MatricaKonfuzije(2);
        
        for (DataSetRow row : testSet) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            int actual = (int)Math.round(row.getDesiredOutput()[0]);
            int predicted = (int)Math.round(neuralNet.getOutput()[0]);
            cm.incrementMatrixElement(actual, predicted);
        }
        cm.ispisi();
        //racunamo accuracy i vracamo ga ((TP+TN)/TP+TN+FP+FN)
        double accuracy = (double)(cm.getTruePositive(0) + cm.getTrueNegative(0))/cm.total;
        System.out.println("Moj accuracy: " + accuracy);
        return accuracy;
    }

    private void saveNetWithMaxAccuracy() {
        Training max = trainings.get(0);
        for (int i = 0; i < trainings.size(); i++) {
            if (trainings.get(i).getAccuracy() > max.getAccuracy()) {
                max = trainings.get(i);
            }
        }
        max.getNeuralNetwork().save("nn2.nnet");
    }

}
