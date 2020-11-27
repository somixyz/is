/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetes;


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
public class Diabetes implements LearningEventListener{
    
    int inputCount = 8; 
    int outputCount = 1;
    double[] learRate = {0.2, 0.4, 0.6}; 
    List<Training> trainings = new ArrayList<>();
    
    public static void main(String[] args) {
        (new Diabetes()).run();
    }
    
    
    public void run() {
        
        String dataSetFile = "diabetes_data.csv";
        //prvo ucitavamo podatke iz fajla, prvo ide putanja, 
        //potom inputcount, pa outputCount, separator na kraju(u zavisnosti od toga kakav je DS)
        //zapeta je separator za CSV, a za txt fajlove vidite u fajlu
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputCount, outputCount, ",");
        
        //normalizacija podataka
        Normalizer norm = new MaxNormalizer(dataSet);
        norm.normalize(dataSet);
        dataSet.shuffle(); //mesamo redosled redova u DataSetu
        
        //podelimo podatke u training i test set
        DataSet[] trainAndTest = dataSet.split(0.7, 0.3);
        DataSet trainSet = trainAndTest[0];
        DataSet testSet = trainAndTest[1];
        
        //u zadatku se zahteva da se ispisu srednje vrednosti 
        //broja iteracija potrebnih za trening svih mreza
        int numOfIteration = 0;
        int numOfTrainings = 0;
        
        for(int i = 0; i < learRate.length; i++){
            //kreiramo neuronsku mrezu tipa visesloni perceptron koji prima broj neurona
            //na osnovu kojih uci i broj skrivenih neurona
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, 20, 10, outputCount);
            
            //postavljamo LearningRule - algoritam koji nam je receno da koristimo
            //BackPropagation ili MomentumBackPropagation i podesavamo mu momentum/gresku po zadatku
            BackPropagation learningRule = neuralNet.getLearningRule();
            //ovde dodajemo Listener - ne trazi se da ga ubacimo, sluzi da pratimo kako mreza uci
            learningRule.addListener(this);
            
            //sada mu setujem learningRate i max gresku
            learningRule.setLearningRate(learRate[i]);
            learningRule.setMaxError(0.07);
            //sada ucimo nasu mrezu, uci se nad trainingSetom
            neuralNet.learn(trainSet);
            //OVDE JE MREZA ZAVRSILA UCENJE
            
            //srednji broj iteracija
            numOfTrainings++;
            numOfIteration += learningRule.getCurrentIteration();
            
            //posle svakog treninga racunamo accuracy i dodajemo
            //taj trening u nasu listu treninga da bismo posle 
            //videli koji je max accuracy (u kom treningu)
            double accuracy = evaluateAcc(neuralNet, testSet);
            Training t = new Training(neuralNet, accuracy);
            trainings.add(t);                
        }
        System.out.println("Srednja vrednost broja iteracija je: " + (double)numOfIteration/numOfTrainings);
        saveNetWithMaxAccuracy();
    }
    
    //kada nije dozvoljeno koriscenje postojecih klasa i metoda,
    //a imamo jedan output
    private double evaluateAcc(MultiLayerPerceptron neuralNet, DataSet test) {
        //pravimo matricu konfuzije sa 2 classCounta
        MatricaKonfuzije cm = new MatricaKonfuzije(2);
        
        //za svaki red u test setu
        for (DataSetRow row : test) {
            //setujemo input za neuronsku mrezu
            //i radimo kalkulaciju nad celom mrezom
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            //uzimamo vrednost za stvarne i predvidjene vrednosti
            //ovde nam niz ima samo 1 element, jer ima 1 output
            //pa uzimamo [0] odnosno taj prvi i jedini element
            int actual = (int)Math.round(row.getDesiredOutput()[0]);
            int predicted = (int)Math.round(neuralNet.getOutput()[0]);
            
            
            //punimo matricu brojevima
            cm.incrementMatrixElement(actual, predicted);
        }
        
        cm.ispisi();
        
        //racunamo accuracy i vracamo ga ((TP+TN)/TP+TN+FP+FN)
        double accuracy = (double)(cm.getTruePositive(0) + cm.getTrueNegative(0))/cm.total;
        System.out.println("Moj accuracy: " + accuracy);
        return accuracy;
    }
    
    
    public void saveNetWithMaxAccuracy(){
        Training max = trainings.get(0);
        for(int i = 1; i < trainings.size(); i++){
            if(trainings.get(i).getAccuracy()> max.getAccuracy())
                max = trainings.get(i);
        }
        max.getNeuralNet().save("nn1.nnet");
   }
    
    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration() + " Total network error: " + bp.getTotalNetworkError());
        //za trenutnu iteraciju prikazuje nam totalnu gresku
        //kada pokrenemo, vidimo da se jedan trening prekine, 
        //kada se greska u sledecoj iteraciji ne promeni
    }        
    
    //metoda za racunanje acc, dozvoljeno koriscenje postojecih klasa,
    //za slucaj kada imamo jedan output
    public double evaluateAccFinishedClasses(MultiLayerPerceptron neuralNet, DataSet testSet) {
        Evaluation evaluation = new Evaluation();
        
        //dodajemo 0.5 u evaluaciju, koristimo Binary jer imamo jedan output
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, testSet); //izvrsava se evaluacija
        
        //getujemo evaluator koji smo napravili gore
        //napravimo nov ConfussionMatrix sa rezultatom
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.Binary.class);
        ConfusionMatrix confusionMatrix = evaluator.getResult();
        
        //prikazemo matricu konfuzije
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        
        //nakon prikazivanja matrice konfuzije, prikazujemo sve metrike
        //i vadimo accuracy i returnujemo ga
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        for(ClassificationMetrics cm : metrics){
            System.out.println(cm.toString() + "\r\n");
        }
     
        System.out.println("ACCURACY: " + average.accuracy);
        double acc = average.accuracy;
        return acc;      
    }
    
}

    

    
    

