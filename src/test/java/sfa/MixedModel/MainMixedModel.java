package sfa.MixedModel;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
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

@RunWith(JUnit4.class)
public class MainMixedModel {

    // The multivariate datasets to use
    public static String[] datasets = new String[]{
            "WISDM-6C-MIXED",
    };

    @Test
    public void testMixedModelClassification() throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/").getFile());

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {


                    int num_sources = 3;
                    int segment_length = 200;
                    Classifier.DEBUG = DEBUG;
                    TimeSeriesLoader.DEBUG = DEBUG;


                    File trainFile = new File(dir.getAbsolutePath() + "/" + s + "/" + "/TRAIN");


                    MultiVariateTimeSeries[] allSamples = TimeSeriesLoader.loadMultivariateDataset(trainFile, num_sources, segment_length);


                    int folds = 10;
                    int[][] testIndices = new int[folds][];
                    int[][] trainIndices = new int[folds][];

                    Classifier.generateIndicesStatic(allSamples, folds, trainIndices, testIndices);

                    double somaAccuracy = 0;
                    double somaPrecision = 0;
                    double somaRecall = 0;
                    double somafmeasure = 0;

                    for (int f = 0; f < folds; f++) {

                        MultiVariateTimeSeries[] trainSamples = getSamplesUsingIndes(allSamples, trainIndices[f]);
                        MultiVariateTimeSeries[] testSamples = getSamplesUsingIndes(allSamples, testIndices[f]);
                        BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                        Classifier.Score result = stack.eval(trainSamples, testSamples);

                        somaAccuracy += result.confusionMatrix.getAccuracy();
                        somaPrecision += result.confusionMatrix.getAvgPrecision();
                        somaRecall += result.confusionMatrix.getAvgRecall();
                        somafmeasure += result.confusionMatrix.getMacroFMeasure();
                        System.out.println(result.confusionMatrix.getAccuracy());
                        if (DEBUG) {
                            System.out.println(result.outputString);
                        }

                    }


                    DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                    otherSymbols.setDecimalSeparator('.');
                    DecimalFormat df = new DecimalFormat("#.00", otherSymbols);
                    System.out.println(s + "," + trainFile.getName() + "," + num_sources + "," + allSamples.length + "," + df.format(somaAccuracy / folds * 100) + "," + df.format(somaPrecision / folds * 100) + "," + df.format(somaRecall / folds * 100) + "," + df.format(somafmeasure / folds * 100));

                }

            }


        } finally

        {
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
