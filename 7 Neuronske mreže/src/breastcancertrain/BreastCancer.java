/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package breastcancertrain;

import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author acer e1
 */
public class BreastCancer implements LearningEventListener {

    int inputCount = 30, outputCount = 1;

    double[] learningRate = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {10, 20, 30};
    List<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        new BreastCancer().run();
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
    }

    private void run() {
        String dataSetFile = "breast_cancer_data.csv";
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputCount, outputCount, ",");

        Normalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();

        DataSet[] trainAndTest = dataSet.split(0.65, 0.35);
        DataSet trainSet = trainAndTest[0];
        DataSet testSet = trainAndTest[0];

        int numOfIteration = 0;
        int numOfTraining = 0;

        for (int i = 0; i < learningRate.length; i++) {
            for (int j = 0; j < hiddenNeurons.length; j++) {

                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hiddenNeurons[j], outputCount);

                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

                learningRule.addListener(this);

                learningRule.setLearningRate(learningRate[i]);
                learningRule.setMomentum(0.7);
                neuralNet.learn(trainSet);

                numOfTraining++;
                numOfIteration += learningRule.getCurrentIteration();

                double mse = evaluateMSE(neuralNet, testSet);
                Training t = new Training(neuralNet, mse);
                trainings.add(t);
            }

            System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIteration / numOfTraining);
            saveNetWithMinError();

        }

    }
    //primer kako se MSE racuna kada nije dozvoljeno koriscenje postojecih klasa i metoda,
    //u slucaju kada imamo jednu output varijablu
    private double evaluateMSE(MultiLayerPerceptron neuralNet, DataSet testSet) {
        double sumError = 0, mse;
        
        for (DataSetRow dataSetRow : testSet) {
            neuralNet.setInput(dataSetRow.getInput());
            neuralNet.calculate();
        
        double[] actual = dataSetRow.getDesiredOutput();
        double[] predicted = neuralNet.getOutput();
        
        sumError += (double) Math.pow((actual[0]-predicted[0]), 2);
        }
        
        mse = (double) sumError/(2*testSet.size());
        System.out.println("Srednja kvadratna greska: "+mse);
        return mse;
    }

    private void saveNetWithMinError() {
        Training min = trainings.get(0);
        for (int i = 0; i < trainings.size(); i++) {
            if (min.getError() > trainings.get(i).getError()) {
                min = trainings.get(i);
            }
        }
        min.getNeuralNet().save("nn1.nnet");
    }
    
    //primer kako se racuna MSE kada je dozvoljeno koriscenje postojecih klasa i metoda,
    //tj klase Evaluation i njenih metoda, a imamo jednu output varijablu
    private double evaluateMSEFinishedClassesJedanOutput(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        //prosledjujem 0.5 jer je to THRESHOLD
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5)); 
        /*
        binarne - imam jednu izlaznu varijablu
        ordinalne - koje se porede, kada gledamo sorte vina, imamo vise varijabli output
        */
        evaluation.evaluate(neuralNet, test); //izvrsava se evaluacija
        
        System.out.println("\nSrednja kvadratna greska(MSE): "+ evaluation.getMeanSquareError() + "\n");
        return evaluation.getMeanSquareError();
    }
    
    //metoda za racunanje mse, nije dozvoljeno koriscenje postojecih klasa,
    //za slucaj kada ima vise outputa
    private double evaluateMSEViseOutputa(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumMse=0, mse;
        for (DataSetRow row : test) {
            //setujemo input za neuronsku mrezu
            //i radimo kalkulaciju nad celom mrezom
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            //uzimamo vrednost za stvarne i predvidjene vrednosti
            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();
            
            //za vise outputa prolazimo kroz niz outputa
            //i dodajemo gresku u sumu
            for (int i = 0; i < actual.length; i++) {
                sumMse += (double)Math.pow((actual[i] - predicted[i]), 2);
            }
        }
        //ovako se racuna totalError u Evaluation -> MeanSquaredError klasi
        mse = (double)sumMse/(2*test.size());
        System.out.println("Moj mse: " + mse);
        return mse;
    }
    
    //metoda za racunanje mse, dozvoljeno koriscenje postojecih klasa,
    //za slucaj kada ima vise outputa
    private void evaluateMSEFinishedClassesViseOutputa(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        //zavisi koliko outputa imamo
        String[] outClasses = {"c1","c2","c3","c4","c5","c6","c7"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(outClasses));
        evaluation.evaluate(neuralNet, test);
        
        double mse = evaluation.getMeanSquareError();
        System.out.println("\nNjihov mse: " + mse + "\n");
    }
    
    
    
}
