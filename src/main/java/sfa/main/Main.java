package sfa.main;

import sfa.classification.BOSSMDStackClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

public class Main {

    public static void main (String[] args){
        String[] p_datasets_subjectIndependent = new String[]{
                "WISDM-MDI",
                "UCI-MDI-OVER",
                "UniMiB-MDI",
        };
        String[] p_segmentLenght = new String[]{
                "200",
                "128",
                "151",
        };
        String[] p_datasets_subjectDependent = new String[]{
                "WISDM-MDU",
                "UCI-MDU-OVER",
                "UniMiB-MDU",
        };

        int n_source = Integer.valueOf(args[1]);

        try {
            switch (Integer.valueOf(args[0])) {
                case 0:
                    testSubjectIndependentClassification(p_datasets_subjectIndependent, p_segmentLenght, n_source);
                case 1:
                    testSubjectDependentClassification(p_datasets_subjectDependent, p_segmentLenght, n_source);
                case 2:

                default:
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        ParallelFor.shutdown();
    }

    public static void testSubjectIndependentClassification(String[] datasets_subjectIndependent, String[] segmentLenght, int n_source) throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/leave-subject-out";

            File dir = new File(home);

            for(int parm = 0; parm < datasets_subjectIndependent.length; parm++) {
                File d = new File(dir.getAbsolutePath() + "/" + datasets_subjectIndependent[parm]);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {
                    double somaAccuracy = 0;
                    double somaPrecision = 0;
                    double somaRecall = 0;
                    double somafmeasure = 0;

                    int num_sources = n_source;
                    int segment_length = Integer.valueOf(segmentLenght[parm]);
                    Classifier.DEBUG = DEBUG;
                    TimeSeriesLoader.DEBUG = DEBUG;
                    int countUsers = 0;
                    for (File userFile : d.listFiles()) {
                        if (userFile.exists() && userFile.isDirectory()) {

                            File trainFile = new File(dir.getAbsolutePath() + "/" + datasets_subjectIndependent[parm] + "/" + userFile.getName() + "/TRAIN");
                            File testFile = new File(dir.getAbsolutePath() + "/" + datasets_subjectIndependent[parm] + "/" + userFile.getName() + "/TEST");



                            MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(trainFile, num_sources, segment_length);
                            MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(testFile, num_sources, segment_length);


                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples, testSamples);

                            somaAccuracy += result.confusionMatrix.getAccuracy();
                            somaPrecision += result.confusionMatrix.getAvgPrecision();
                            somaRecall += result.confusionMatrix.getAvgRecall();
                            somafmeasure += result.confusionMatrix.getMacroFMeasure();
                            countUsers++;
                            System.out.print(userFile.getName() + ',');
                            System.out.print(String.valueOf(result.confusionMatrix.getAccuracy()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getAvgPrecision()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getAvgRecall()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getMacroFMeasure()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getConfidence95Accuracy()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getConfidence95MacroFM()));
                            System.out.println();
                            if (DEBUG) {
                                System.out.println(result.outputString);
                            }
                            stack.shutdown();
                        }


                    }
                    DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                    otherSymbols.setDecimalSeparator('.');
                    DecimalFormat df = new DecimalFormat("#.00", otherSymbols);
                    System.out.println(datasets_subjectIndependent[parm] + "," + d.getName() + "," + num_sources + "," +  df.format(somaAccuracy / countUsers *100 ) + "," + df.format(somaPrecision / countUsers *100) + "," + df.format(somaRecall / countUsers *100) + "," + df.format(somafmeasure / countUsers *100));
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    public static void testSubjectDependentClassification(String[] datasets_subjectDependent, String[] lenghts, int n_source) throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/cross-validation-subject";

            File dir = new File(home);

            for(int parm_index = 0; parm_index < datasets_subjectDependent.length; parm_index++) {
                File d = new File(dir.getAbsolutePath() + "/" + datasets_subjectDependent[parm_index]);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {

                    //each USER
                    for (File train : d.listFiles()) {

                        int num_sources = n_source;
                        int segment_length = Integer.valueOf(lenghts[parm_index]);
                        String filename = train.getName();
                        Classifier.DEBUG = DEBUG;
                        TimeSeriesLoader.DEBUG = DEBUG;
                        MultiVariateTimeSeries[] allSamples = TimeSeriesLoader.loadMultivariateDataset(train, num_sources, segment_length);

                        int folds = 10;
                        int[][] testIndices = new int[folds][];
                        int[][] trainIndices = new int[folds][];

                        Classifier.generateIndicesStatic(allSamples, folds,trainIndices,testIndices);

                        double somaAccuracy = 0;
                        double somaPrecision = 0;
                        double somaRecall= 0;
                        double somafmeasure = 0;

                        for (int f = 0; f < folds; f++){

                            MultiVariateTimeSeries[] trainSamples = getSamplesUsingIndex(allSamples,trainIndices[f]);
                            MultiVariateTimeSeries[] testSamples = getSamplesUsingIndex(allSamples,testIndices[f]);
                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples,testSamples);

                            somaAccuracy += result.confusionMatrix.getAccuracy();
                            somaPrecision += result.confusionMatrix.getAvgPrecision();
                            somaRecall += result.confusionMatrix.getAvgRecall();
                            somafmeasure += result.confusionMatrix.getMacroFMeasure();
                            System.out.print(train.getName() + ',');
                            System.out.print(String.valueOf(result.confusionMatrix.getAccuracy()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getAvgPrecision()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getAvgRecall()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getMacroFMeasure()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getConfidence95Accuracy()) + ",");
                            System.out.print(String.valueOf(result.confusionMatrix.getConfidence95MacroFM()));
                            System.out.println();
                            stack.shutdown();

                        }
                        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                        otherSymbols.setDecimalSeparator('.');
                        DecimalFormat df = new DecimalFormat("#.00",otherSymbols);
                        System.out.println(datasets_subjectDependent[parm_index] + "," + filename + "," + num_sources + "," + allSamples.length + "," + df.format(somaAccuracy/ folds * 100) + "," + df.format(somaPrecision/ folds * 100) + "," + df.format(somaRecall/ folds * 100) + "," + df.format(somafmeasure/ folds * 100));
                    }
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    private static MultiVariateTimeSeries[] getSamplesUsingIndex(MultiVariateTimeSeries[] allSamples, int[] indices) {
        MultiVariateTimeSeries[] result = new MultiVariateTimeSeries[indices.length];
        int count = 0;
        for (int i : indices) {
            result[count] = allSamples[i];
            count++;
        }
        return result;
    }
}
