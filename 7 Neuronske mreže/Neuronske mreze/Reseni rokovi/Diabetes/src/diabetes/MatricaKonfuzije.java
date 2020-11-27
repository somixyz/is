/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diabetes;

import org.neuroph.eval.classification.ConfusionMatrix;

/**
 *
 * @author Ari
 */

//pravila se nova klasa MatricaKonfuzije, u kojoj su dodate neke od metoda 
//koje se nalaze u postojecoj klasi ConfusionMatrix
//najlakse je da kada se napravi nova klasa, otvorite postojecu klasu ConfusionMatrix
//i prekopirae metode koje su neophodne i u njima samo izmenite nazive parametara
//na taj nacin niko ne bi smarao da ste kopirali postojece metode
//neophodne metode su ove koje se nalaze klasi MatricaKonfuzije ispod, 
//jedina koje je bas rucno od nule pisana je metoda koja ispisuje izgled matrice,
//dakle nije prekopiran ToString iz ConfusionMatrix

class MatricaKonfuzije {
    
    int classCount;
    int[][] matrix;
    int total = 0;

    public MatricaKonfuzije(int classCount) {
        this.classCount = classCount;
        this.matrix = new int[classCount][classCount];
    }

    public void incrementMatrixElement(int actual, int predicted) {
        matrix[actual][predicted]++;
        total++;
    }

    public int getTruePositive(int klasa) {
        return (int) matrix[klasa][klasa];
    }

    public int getTrueNegative(int klasa) {
        int trueNegative = 0;

        for (int i = 0; i < classCount; i++) {
            if (i == klasa) {
                continue;
            }
            for (int j = 0; j < classCount; j++) {
                if (j == klasa) {
                    continue;
                }
                trueNegative += matrix[i][j];
            }
        }

        return trueNegative;
    }

    public int getFalsePositive(int klasa) {
        int falsePositive = 0;

        for (int i = 0; i < classCount; i++) {
            if (i == klasa) {
                continue;
            }
            falsePositive += matrix[i][klasa];
        }

        return falsePositive;
    }

    public int getFalseNegative(int klasa) {
        int falseNegative = 0;

        for (int i = 0; i < classCount; i++) {
            if (i == klasa) {
                continue;
            }
            falseNegative += matrix[klasa][i];
        }
        return falseNegative;
    }

    public void ispisi() {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}
