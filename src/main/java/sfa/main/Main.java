package sfa.main;

import sfa.classification.BOSSMDStackClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.metrics.ConfusionMatrix;
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
                "WISDM-MDI-X",
                "UCI-MDI-OVER",
                "UniMiB-MDI",
        };
        String[] p_segmentLenght = new String[]{
                "200",
                "128",
                "151",
        };
        String[] p_datasets_subjectDependent = new String[]{
                "WISDM-MDU-6C",
                "UCI-MDU-OVER",
                "UniMiB-MDU",
        };
        String[] p_datasets_21e9 = new String[]{
                "UCI",

        };


        //int n_source = Integer.valueOf(args[1]);

        try {
            /*switch (Integer.valueOf(args[0])) {
                case 0:
                    testSubjectIndependentClassification(p_datasets_subjectIndependent, p_segmentLenght, n_source);
                case 1:
                    testSubjectDependentClassification(p_datasets_subjectDependent, p_segmentLenght, n_source);
                case 2:
                    test21e9Classification(p_datasets_21e9, p_segmentLenght, n_source);
                default:
                    timeProcessing();
            }*/
            timeProcessing();
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
                System.out.println("==========================");
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
                    ConfusionMatrix cmGeral = new ConfusionMatrix();
                    for (File userFile : d.listFiles()) {
                        if (userFile.exists() && userFile.isDirectory()) {

                            File trainFile = new File(dir.getAbsolutePath() + "/" + datasets_subjectIndependent[parm] + "/" + userFile.getName() + "/TRAIN");
                            File testFile = new File(dir.getAbsolutePath() + "/" + datasets_subjectIndependent[parm] + "/" + userFile.getName() + "/TEST");



                            MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(trainFile, num_sources, segment_length);
                            MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(testFile, num_sources, segment_length);


                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples, testSamples);
                            System.out.println(result.outputString);
                            cmGeral = ConfusionMatrix.createCumulativeMatrix(cmGeral,result.confusionMatrix);

                            stack.shutdown();
                        }


                    }
                    System.out.println("==========================");
                    System.out.println("dataset,Accuracy,Precision,Recall,Fscore,95Accuracy,95Fscore,Kappa");
                    System.out.print(datasets_subjectIndependent[parm] + ',');
                    System.out.print(String.valueOf(cmGeral.getAccuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgPrecision()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgRecall()) + ",");
                    System.out.print(String.valueOf(cmGeral.getMacroFMeasure()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95Accuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95MacroFM()) + ",");
                    System.out.println(String.valueOf(cmGeral.getCohensKappa()));
                    System.out.println("==========================");
                    System.out.println(cmGeral.printClassDistributionGold());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringLatex());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toString());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringProbabilistic());
                    System.out.println("==========================");
                    System.out.println(" ");
                    cmGeral.toExportToFile(System.getProperty("user.home") + "/logs/confusionmatrix/leave-subject-out/" ,datasets_subjectIndependent[parm]);

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
                System.out.println("==========================");


                if (d.exists() && d.isDirectory()) {

                    //each USER
                    ConfusionMatrix cmGeral = new ConfusionMatrix();
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
                            System.out.println(result.outputString);
                            cmGeral = ConfusionMatrix.createCumulativeMatrix(cmGeral,result.confusionMatrix);

                            stack.shutdown();

                        }


                    }
                    System.out.println("==========================");
                    System.out.println("dataset,Accuracy,Precision,Recall,Fscore,95Accuracy,95Fscore,Kappa");
                    System.out.print(datasets_subjectDependent[parm_index] + ',');
                    System.out.print(String.valueOf(cmGeral.getAccuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgPrecision()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgRecall()) + ",");
                    System.out.print(String.valueOf(cmGeral.getMacroFMeasure()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95Accuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95MacroFM()) + ",");
                    System.out.println(String.valueOf(cmGeral.getCohensKappa()));
                    System.out.println("==========================");
                    System.out.println(cmGeral.printClassDistributionGold());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringLatex());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toString());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringProbabilistic());
                    System.out.println("==========================");
                    System.out.println(" ");
                    cmGeral.toExportToFile(System.getProperty("user.home") + "/logs/confusionmatrix/cross-validation-subject/" ,datasets_subjectDependent[parm_index]  );
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    public static void test21e9Classification(String[] datasets, String[] segmentLenght, int n_source) throws IOException{
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/21e9";

            File dir = new File(home);

            for(int parm = 0; parm < datasets.length; parm++) {
                File d = new File(dir.getAbsolutePath() + "/" + datasets[parm]);
                System.out.println("==========================");
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




                    File trainFile = new File(dir.getAbsolutePath() + "/" + datasets[parm] +  "/TRAIN");
                    File testFile = new File(dir.getAbsolutePath() + "/" + datasets[parm] + "/TEST");



                    MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(trainFile, num_sources, segment_length);
                    MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(testFile, num_sources, segment_length);


                    BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                    Classifier.Score result = stack.eval(trainSamples, testSamples);
                    System.out.println(result.outputString);
                    ConfusionMatrix cmGeral = result.confusionMatrix;

                    stack.shutdown();




                    System.out.println("==========================");
                    System.out.println("dataset,Accuracy,Precision,Recall,Fscore,95Accuracy,95Fscore,Kappa");
                    System.out.print(datasets[parm] + ',');
                    System.out.print(String.valueOf(cmGeral.getAccuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgPrecision()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgRecall()) + ",");
                    System.out.print(String.valueOf(cmGeral.getMacroFMeasure()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95Accuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95MacroFM()) + ",");
                    System.out.println(String.valueOf(cmGeral.getCohensKappa()));
                    System.out.println("==========================");
                    System.out.println(cmGeral.printClassDistributionGold());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringLatex());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toString());
                    System.out.println("==========================");
                    System.out.println(cmGeral.toStringProbabilistic());
                    System.out.println("==========================");
                    System.out.println(" ");
                    cmGeral.toExportToFile(System.getProperty("user.home") + "/logs/confusionmatrix/21e9/" ,datasets[parm]);

                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    public static void timeProcessing() throws IOException {
        try {


            boolean DEBUG = false;
            String home = System.getProperty("user.home") + "/datasets/dataset1";
            String home2 = System.getProperty("user.home") + "/datasets/dataset2";

            File file = new File(home);
            File file2 = new File(home2);


            if (file.exists()) {
                int num_sources = 3;
                int segment_length = 151;
                Classifier.DEBUG = DEBUG;
                TimeSeriesLoader.DEBUG = DEBUG;

                MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(file2, num_sources, segment_length);
                MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(file, num_sources, segment_length);

                BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                Classifier.Score result = stack.eval(trainSamples, testSamples);
                System.out.println(result.outputString);


                stack.shutdown();
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
