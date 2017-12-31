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
import java.util.ArrayList;
import java.util.Locale;

public class Main {

    public static void main (String[] args){
        String[] datasets = new String[]{
                //"WISDM-MDI",
                //"UCI-MDI",
                "UniMiB-MDI",
        };
        /*ArrayList<String> list = new ArrayList<>();
        for(int i = 0; i < args.length; i++){
            list.add(args[i]);
        }*/
        try {
            testSubjectIndependentClassification(datasets);
            //testSubjectDependentClassification(list.toArray(new String[]{}));
        } catch (IOException e) {
            e.printStackTrace();
        }
        ParallelFor.shutdown();
    }

    public static void testSubjectIndependentClassification(String[] datasets) throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/leave-subject-out";

            File dir = new File(home);

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {
                    double somaAccuracy = 0;
                    double somaPrecision = 0;
                    double somaRecall = 0;
                    double somafmeasure = 0;

                    int num_sources = 3;
                    int segment_length = 151;
                    Classifier.DEBUG = DEBUG;
                    TimeSeriesLoader.DEBUG = DEBUG;
                    int countUsers = 0;
                    for (File userFile : d.listFiles()) {
                        if (userFile.exists() && userFile.isDirectory()) {

                            File trainFile = new File(dir.getAbsolutePath() + "/" + s + "/" + userFile.getName() + "/TRAIN");
                            File testFile = new File(dir.getAbsolutePath() + "/" + s + "/" + userFile.getName() + "/TEST");



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
                            System.out.print(String.valueOf(result.confusionMatrix.getConfidence95MacroFM()) + ",");
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
                    System.out.println(s + "," + d.getName() + "," + num_sources + "," +  df.format(somaAccuracy / countUsers *100 ) + "," + df.format(somaPrecision / countUsers *100) + "," + df.format(somaRecall / countUsers *100) + "," + df.format(somafmeasure / countUsers *100));
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    public static void testSubjectDependentClassification(String[] datasets) throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/cross-validation-subject";

            File dir = new File(home);

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {

                    //each USER
                    for (File train : d.listFiles()) {

                        int num_sources = 3;
                        int segment_length = 151;
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

                            MultiVariateTimeSeries[] trainSamples = getSamplesUsingIndes(allSamples,trainIndices[f]);
                            MultiVariateTimeSeries[] testSamples = getSamplesUsingIndes(allSamples,testIndices[f]);
                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples,testSamples);

                            somaAccuracy += result.confusionMatrix.getAccuracy();
                            somaPrecision += result.confusionMatrix.getAvgPrecision();
                            somaRecall += result.confusionMatrix.getAvgRecall();
                            somafmeasure += result.confusionMatrix.getMacroFMeasure();
                            if(DEBUG) {
                                System.out.println("Accuracy:" + result.confusionMatrix.getAccuracy());
                                System.out.println("AvgPrecision:" + result.confusionMatrix.getAvgPrecision());
                                System.out.println("AvgRecall:" + result.confusionMatrix.getAvgRecall());
                                System.out.println("MacroFmeasure:" + result.confusionMatrix.getMacroFMeasure());
                            }

                        }
                        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                        otherSymbols.setDecimalSeparator('.');
                        DecimalFormat df = new DecimalFormat("#.00",otherSymbols);
                        System.out.println(s + "," + filename + "," + num_sources + "," + allSamples.length + "," + df.format(somaAccuracy/ folds * 100) + "," + df.format(somaPrecision/ folds * 100) + "," + df.format(somaRecall/ folds * 100) + "," + df.format(somafmeasure/ folds * 100));
                    }
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    private static MultiVariateTimeSeries[] getSamplesUsingIndes(MultiVariateTimeSeries[] allSamples, int[] indices) {
        MultiVariateTimeSeries[] result = new MultiVariateTimeSeries[indices.length];
        int count = 0;
        for (int i : indices) {
            result[count] = allSamples[i];
            count++;
        }
        return result;
    }
}
