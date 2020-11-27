/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package glass;

/**
 *
 * @author Ari
 */
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

    public void Ispisi() {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }
}
