/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wine;

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
 * @author Ari
 */
//uvek implementiramo LearningEventListener
public class Wines implements LearningEventListener {

    int inputCount = 13;
    int outputCount = 3;
    double[] learRate = {0.2, 0.4, 0.6};
    List<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        (new Wines()).run();
    }

    public void run() {
        String dataSetFile = "wines.csv";
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputCount, outputCount, ",");

        //normalizacija podataka
        Normalizer norm = new MaxNormalizer(dataSet);
        norm.normalize(dataSet);
        dataSet.shuffle(); //mesamo redosled redova u DataSetu

        DataSet[] trainAndTest = dataSet.split(0.7, 0.3);
        DataSet trainSet = trainAndTest[0];
        DataSet testSet = trainAndTest[1];
        int numOfIteration = 0;
        int numOfTrainings = 0;
        for (int i = 0; i < learRate.length; i++) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, 22, outputCount);

            BackPropagation learningRule = neuralNet.getLearningRule();
            learningRule.addListener(this);
            learningRule.setLearningRate(learRate[i]);
            learningRule.setMaxError(0.02);
            neuralNet.learn(trainSet);
            numOfTrainings++;
            numOfIteration += learningRule.getCurrentIteration();

            double accuracy = evaluateAcc(neuralNet, testSet);
            Training t = new Training(neuralNet, accuracy);
            trainings.add(t);
        }
        System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIteration / numOfTrainings);
        saveNetWithMaxAccuracy();
    }

    //kada nije dozvoljeno koriscenje postojecih klasa i metoda,
    //a imamo vise outputa
    private double evaluateAcc(MultiLayerPerceptron neuralNet, DataSet test) {
        MatricaKonfuzije cm = new MatricaKonfuzije(outputCount);
        double accuracy = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(neuralNet.getOutput());
            cm.incrementMatrixElement(actual, predicted);

        }
        //racunamo accuracy ((TP+TN)/TP+TN+FP+FN))
        for (int i = 0; i < outputCount; i++) {
            accuracy += (double) (cm.getTruePositive(i) + cm.getTrueNegative(i)) / cm.total;
        }
        cm.Ispisi();
        System.out.println("Moj accuracy: " + (double) accuracy / outputCount);
        return (double) accuracy / outputCount;
    }

    private int getMaxIndex(double[] output) {
        int max = 0;
        for (int i = 0; i < output.length; i++) {
            if (output[max] < output[i]) {
                max = i;
            }
        }
        return max;
    }

    public void saveNetWithMaxAccuracy() {
        Training max = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (trainings.get(i).getAccuracy() > max.getAccuracy()) {
                max = trainings.get(i);
            }
        }
        max.getNeuralNet().save("nn1.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        MomentumBackpropagation bp = (MomentumBackpropagation) le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration()
                + " Total network error: " + bp.getTotalNetworkError());
    }

    public double evaluateAccFinishedClasses(MultiLayerPerceptron neuralNet, DataSet testSet) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = new String[]{"c1", "c2", "c3"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, testSet); //izvrsava se evaluacija
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix cm = evaluator.getResult();

        System.out.println(cm.toString() + "\r\n\r\n");

        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics classMet : metrics) {
            System.out.println(classMet.toString() + "\r\n");
        }
        System.out.println("Accuracy: " + average.accuracy);
        return average.accuracy;

    }

}
