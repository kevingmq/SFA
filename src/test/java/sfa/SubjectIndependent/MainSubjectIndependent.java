package sfa.SubjectIndependent;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
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

@RunWith(JUnit4.class)
public class MainSubjectIndependent {

    // The multivariate datasets to use
    public static String[] datasets = new String[]{
            "WISDM-MDI",
            //"UCI-MDI",
            //"UniMiB-MDI",
    };

    @Test
    public void testSubjectIndependentClassification() throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/").getFile());

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {
                    double somaAccuracy = 0;
                    double somaPrecision = 0;
                    double somaRecall = 0;
                    double somafmeasure = 0;

                    int num_sources = 3;
                    int segment_length = 200;
                    Classifier.DEBUG = DEBUG;
                    TimeSeriesLoader.DEBUG = DEBUG;
                    int countUsers = 0;

                    ConfusionMatrix cmGeral = new ConfusionMatrix();

                    for (File userFile : d.listFiles()) {
                        if (userFile.exists() && userFile.isDirectory()) {

                            File trainFile = new File(dir.getAbsolutePath() + "/" + s + "/" + userFile.getName() + "/TRAIN");
                            File testFile = new File(dir.getAbsolutePath() + "/" + s + "/" + userFile.getName() + "/TEST");



                            MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(trainFile, num_sources, segment_length);
                            MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDataset(testFile, num_sources, segment_length);


                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples, testSamples);
                            ConfusionMatrix atual = ConfusionMatrix.createCumulativeMatrix(cmGeral,result.confusionMatrix);
                            cmGeral = atual;
                            //somaAccuracy += result.confusionMatrix.getAccuracy();
                            //somaPrecision += result.confusionMatrix.getAvgPrecision();
                            //somaRecall += result.confusionMatrix.getAvgRecall();
                            //somafmeasure += result.confusionMatrix.getMacroFMeasure();
                            //countUsers++;
                            if (DEBUG) {
                                System.out.println(result.outputString);
                            }

                        }


                    }
                    DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                    otherSymbols.setDecimalSeparator('.');
                    DecimalFormat df = new DecimalFormat("#.00", otherSymbols);
                    System.out.println(s + "," + d.getName() + "," + num_sources + "," +  df.format(somaAccuracy / countUsers *100 ) + "," + df.format(somaPrecision / countUsers *100) + "," + df.format(somaRecall / countUsers *100) + "," + df.format(somafmeasure / countUsers *100));
                    System.out.print(String.valueOf(cmGeral.getAccuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgPrecision()) + ",");
                    System.out.print(String.valueOf(cmGeral.getAvgRecall()) + ",");
                    System.out.print(String.valueOf(cmGeral.getMacroFMeasure()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95Accuracy()) + ",");
                    System.out.print(String.valueOf(cmGeral.getConfidence95MacroFM())+ ",");
                    System.out.print(String.valueOf(cmGeral.getCohensKappa()));
                    System.out.println();
                    System.out.println(cmGeral.printClassDistributionGold());
                    System.out.println(cmGeral.toStringProbabilistic());
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    private MultiVariateTimeSeries[] getSamplesUsingIndes(MultiVariateTimeSeries[] allSamples, int[] indices) {
        MultiVariateTimeSeries[] result = new MultiVariateTimeSeries[indices.length];
        int count = 0;
        for (int i : indices) {
            result[count] = allSamples[i];
            count++;
        }
        return result;
    }

}
