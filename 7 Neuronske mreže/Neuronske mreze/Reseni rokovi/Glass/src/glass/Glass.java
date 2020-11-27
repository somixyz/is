/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package glass;

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
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Ari
 */
//uvek implementiramo LearningEventListener
public class Glass implements LearningEventListener{
    
    int inputCount = 9;
    int outputCount = 7;
    double[] learRate = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {10, 20, 30};
    List<Training> trainings = new ArrayList<>();
    
    public static void main(String[] args) {
        (new Glass()).run();
    }

    public void run() {
        String dataSetFile = "glass.csv";
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
                learningRule.setMomentum(0.6);
                learningRule.setMaxIterations(1000); //ovo sam uradio jer se program predugo izvrsava...
                //sada ucimo nasu mrezu, uci se nad trainingSetom
                neuralNet.learn(trainSet);
                //OVDE JE MREZA ZAVRSILA UCENJE
            
                //srednji broj iteracija
                numOfTrainings++;
                numOfIteration += learningRule.getCurrentIteration();
                
                //posle svakog treninga racunamo accuracy i dodajemo
                //taj trening u nasu listu treninga da bismo posle 
                //videli koji je max accuracy (u kom treningu)
                double accuracy = evaluateAccFinishedClasses(neuralNet, testSet);
                Training t = new Training(neuralNet, accuracy);
                trainings.add(t);
            }
        }
        System.out.println("Srednja vrednost broja iteracija je: " + (double)numOfIteration/numOfTrainings);
        saveNetWithMaxAccuracy();
        
    }
    
    //kada nije dozvoljeno koriscenje postojecih klasa i metoda,
    //a imamo vise outputa
    private double evaluateAcc(MultiLayerPerceptron neuralNet, DataSet test) {
        MatricaKonfuzije cm = new MatricaKonfuzije(outputCount);
        double accuracy = 0;
        //za svaki red u test setu
        for (DataSetRow row : test) {
            //setujemo input za neuronsku mrezu
            //i radimo kalkulaciju nad celom mrezom
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            //uzimamo vrednost za stvarne i predvidjene vrednosti
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(neuralNet.getOutput());
            
            //punimo matricu brojevima
            cm.incrementMatrixElement(actual, predicted);
            
        }
        
        //racunamo accuracy ((TP+TN)/TP+TN+FP+FN)
        for (int i = 0; i < outputCount; i++) {
            accuracy += (double)(cm.getTruePositive(i) + cm.getTrueNegative(i))/cm.total; 
        }
        
        //ispisemo matricu
        cm.Ispisi();
        
        //ispisemo accuracy i vracamo ga
        System.out.println("Moj accuracy: "+ (double)accuracy/outputCount);
        return (double)accuracy/outputCount;
    }
    
    private int getMaxIndex(double[] output) {
        int max = 0;
        for (int i = 0; i < output.length; i++) {
            if(output[max] < output[i])
                max = i;
        }
        return max;
    }
    
    public void saveNetWithMaxAccuracy() {
       Training max = trainings.get(0);
       for(int i = 1; i < trainings.size(); i++){
           if(trainings.get(i).getAccuracy() > max.getAccuracy())
               max = trainings.get(i);
       }
       max.getNeuralNet().save("nn1.nnet");
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
    }

    //metoda za racunanje acc, dozvoljeno koriscenje postojecih klasa,
    //za slucaj kada imamo vise outputa
    public double evaluateAccFinishedClasses(MultiLayerPerceptron neuralNet, DataSet testSet) {
        Evaluation evaluation = new Evaluation();
        //posto imam vise izlaznih klasa,treba da sacuvam negde njihova imena
        String[] classLabels = {"s1","s2","s3","s4","s5","s6","s7"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        //dobijam matricu 7x7
        evaluation.evaluate(neuralNet,testSet); //izvrsava se evaluacija
        
        //koja klasa treba da bude classifier i uzimamo dobijenu konfuzionu matricu
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class); 
        ConfusionMatrix cm = evaluator.getResult();
        
        //prikazemo matricu konfuzije
        System.out.println(cm.toString());
        
        //nakon prikazivanja matrice konfuzije, prikazujemo sve metrike
        //i vadimo accuracy i returnujemo ga
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        for(ClassificationMetrics classMet : metrics){
            System.out.println(classMet.toString() + "\r\n");
        }
        
        System.out.println("Average accuracy: " + average.accuracy);
        return average.accuracy;
    }
    
}
