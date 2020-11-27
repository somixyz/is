/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package breastcancer;

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
 * @author Ari
 */

//uvek implementiramo LearningEventListener
public class BreastCancer implements LearningEventListener {

    int inputCount = 30;
    int outputCount = 1;
    double[] learRate = {0.2,0.4,0.6};
    int[] hiddenNeurons = {10,20,30};
    List<Training> trainings = new ArrayList<>();
    
    public static void main(String[] args) {
        (new BreastCancer()).run();  
    }
         
    public void run() {
       String dataSetFile = "breast_cancer_data.csv";
       //prvo ucitavamo podatke iz fajla, prvo ide putanja, 
       //potom inputcount, pa outputCount, separator na kraju(u zavisnosti od toga kakav je DS)
       //zapeta je separator za CSV, a za txt fajlove vidite u fajlu
       DataSet dataSet = DataSet.createFromFile(dataSetFile, inputCount, outputCount, ",");
       
        //normalizacija podataka
        Normalizer norm = new MaxNormalizer(dataSet);
        norm.normalize(dataSet);
        dataSet.shuffle(); //mesamo redosled redova u DataSetu
        
        //podelimo podatke u training i test set
        DataSet[] trainAndTest = dataSet.split(0.65, 0.35);
        DataSet trainSet = trainAndTest[0];
        DataSet testSet = trainAndTest[1];
       
        //u zadatku se zahteva da se ispisu srednje vrednosti 
        //broja iteracija potrebnih za trening svih mreza
        int numOfIteration = 0;
        int numOfTrainings = 0;
       
       for(int i = 0; i < learRate.length; i++){
           for(int j = 0; j < hiddenNeurons.length; j++){
               //kreiramo neuronsku mrezu tipa visesloni perceptron koji prima broj neurona
               //na osnovu kojih uci i broj skrivenih neurona
               MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hiddenNeurons[j], outputCount);
               
                //postavljamo LearningRule - algoritam koji nam je receno da koristimo
                //BackPropagation ili MomentumBackPropagation i podesavamo mu momentum/gresku po zadatku
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
                //ovde dodajemo Listener - ne trazi se da ga ubacimo, sluzi da pratimo kako mreza uci
                learningRule.addListener(this);
               
               //sada mu setujem learningRate i momentum
                learningRule.setLearningRate(learRate[i]);
                learningRule.setMomentum(0.7);
                //sada ucimo nasu mrezu, uci se nad trainingSetom
                neuralNet.learn(trainSet);
                //OVDE JE MREZA ZAVRSILA UCENJE
            
                //srednji broj iteracija
                numOfTrainings++;
                numOfIteration += learningRule.getCurrentIteration();
               
               //posle svakog treninga racunamo MSE i dodajemo
               //taj trening u nasu listu treninga da bismo posle
               //videli koji je min MSE (u kom treningu)
               double mse = evaluateMSE(neuralNet, testSet);
               Training t = new Training(neuralNet, mse);
               trainings.add(t);
           }
       }
        System.out.println("Srednja vrednost broja iteracija je: " + (double)numOfIteration/numOfTrainings);
        saveNetWithMinError();
    }
    
    //primer kako se MSE racuna kada nije dozvoljeno koriscenje postojecih klasa i metoda,
    //u slucaju kada imamo jednu output varijablu
    private double evaluateMSE(MultiLayerPerceptron neuralNet, DataSet test) {
        //pravimo ukupnu gresku i srednju gresku
        double sumError = 0, mse;
        
        //za svaki red u test setu
        for (DataSetRow row : test) {
            //setujemo input za neuronsku mrezu
            //i radimo kalkulaciju nad celom mrezom
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            //uzimamo vrednost za stvarne i predvidjene vrednosti
            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();
            
            //sumiramo sve vrednosti greske, ovako se racuna greska
            sumError += (double)Math.pow((actual[0]-predicted[0]),2);
        } 
        //ovako se racuna totalError u Evaluation -> MeanSquaredError klasi
        mse = (double)sumError/(2*test.size());
        System.out.println("Srednja kvadratna greska: " + mse);
        return mse;
    }
    
    public void saveNetWithMinError() {
        Training min = trainings.get(0);
        for(int i = 1; i < trainings.size(); i++){
            if(trainings.get(i).getError() < min.getError())
                min = trainings.get(i);
        }
        min.getNeuralNet().save("nn1.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
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
